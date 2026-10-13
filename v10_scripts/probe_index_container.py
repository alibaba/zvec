# Decode the zvec index container format in pure Python and dump the segment
# table plus the key->list maps needed for the v10 P5 attribution analysis.
#
# Container layout (src/core/framework/index_unpacker.h + index_format.h):
#   [MetaHeader 64B][content ...][SegmentMeta[]x32B][MetaFooter 128B]
# possibly chained via footer.next_meta_header_offset. All offsets absolute.
import mmap
import struct
import sys

HDR_FMT = '<IHHIIHHIQQ3Q'          # 64B
FOOT_FMT = '<6I13Q'                 # 128B
SEGM_FMT = '<IIQQQ'                 # 32B
INVERTED_LIST_FMT = '<QIII'         # 40B stride (align-8)
IVF_HDR_FMT = '<IIQIIIII'          # 36B fields: hdr,total,body,list_cnt,block_vecs,block_sz,block_cnt,meta_sz
IVF_RABITQ_HDR_FMT = '<IIIIIIBBBBQQ4I'  # 60B fields
DED_CLUSTER_FMT = '<QQIIII'         # 32B


class Container:
    def __init__(self, path, base=0, size=None):
        self.f = open(path, 'rb')
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        self.path = path
        self.base = base
        self.total = len(self.mm) - base if size is None else size
        self.segments = {}   # name -> (abs_data_offset, data_size, padding)
        hdr_start = 0
        total = self.total
        while True:
            (h_crc, _r1, ver, rev, magic, mhs, mfs, mfo, content_off,
             setup, _rv1, _rv2, _rv3) = struct.unpack_from(
                 HDR_FMT, self.mm, base + hdr_start)
            assert mhs == 64, f'meta_header_size {mhs} != 64'
            assert mfs == 128, f'meta_footer_size {mfs} != 128'
            mfo_s = struct.unpack_from('<i', self.mm, base + hdr_start + 20)[0]
            foot_off = (total + mfo_s) if mfo_s < 0 else mfo_s
            (f_crc, seg_meta_crc, content_crc, seg_cnt, seg_meta_sz, _fr1,
             content_sz, content_pad, cp, ut, _r21, _r22, _r23, _r24, _r25,
             _r26, _r27, next_hdr, tot_sz) = struct.unpack_from(
                 FOOT_FMT, self.mm, base + hdr_start + foot_off)
            assert seg_meta_sz >= 32 * seg_cnt, \
                f'segments_meta_size {seg_meta_sz} < 32*{seg_cnt}'
            meta_base = base + hdr_start + foot_off - seg_meta_sz
            for i in range(seg_cnt):
                (id_off, data_crc, data_idx, data_sz, pad_sz) = struct.unpack_from(
                    SEGM_FMT, self.mm, meta_base + 32 * i)
                name_b = self.mm[meta_base + id_off:
                                 self.mm.find(b'\x00', meta_base + id_off)]
                name = name_b.decode()
                abs_off = base + hdr_start + content_off + data_idx
                if name not in self.segments:   # std::map::emplace: first wins
                    self.segments[name] = (abs_off, data_sz, pad_sz)
            if next_hdr == 0:
                break
            hdr_start = next_hdr
        self.magic = magic

    def seg(self, name):
        return self.segments[name]

    def seg_data(self, name):
        off, sz, _ = self.segments[name]
        return self.mm[off:off + sz]

    def close(self):
        self.mm.close()
        self.f.close()


def load_int_list_meta(c):
    off, sz, _ = c.seg('ivf.inverted_meta')
    (h_sz, total, body, list_cnt, block_vecs, block_sz, block_cnt,
     meta_sz) = struct.unpack_from(IVF_HDR_FMT, c.seg_data('ivf.inverted_header'))
    assert sz == 40 * list_cnt, \
        f'inverted_meta {sz} != 40*{list_cnt}'
    lists = []
    for l in range(list_cnt):
        blk_off, blk_cnt, vec_cnt, id_off = struct.unpack_from(
            INVERTED_LIST_FMT, c.mm, off + 40 * l)
        lists.append((blk_off, blk_cnt, vec_cnt, id_off))
    return total, list_cnt, block_vecs, block_cnt, lists


def load_ded_cluster_meta(c):
    off, sz, _ = c.seg('ivf_rabitq.cluster_meta')
    (magic, ver, total, cluster_cnt, dim, padded, ex_bits, rot_type,
     metric, _pad, batch_sz, ex_sz, *_r) = struct.unpack_from(
        IVF_RABITQ_HDR_FMT, c.seg_data('ivf_rabitq.header'))
    assert magic == 0x49565251, f'bad magic {magic:#x}'
    assert sz == 32 * cluster_cnt, f'cluster_meta {sz} != 32*{cluster_cnt}'
    clusters = []
    for cl in range(cluster_cnt):
        b_off, e_off, vec_cnt, b_cnt, k_off, _res = struct.unpack_from(
            DED_CLUSTER_FMT, c.mm, off + 32 * cl)
        clusters.append((b_off, e_off, vec_cnt, b_cnt, k_off))
    return total, cluster_cnt, dim, padded, ex_bits, metric, clusters


def build_key_map(c, seg_name, units):
    """key -> (list_id, rank_in_list) from a per-list-concatenated u64 array."""
    off, sz, _ = c.seg(seg_name)
    assert sz == 8 * units, f'{seg_name} size {sz} != 8*{units}'
    keys = struct.unpack_from(f'<{units}Q', c.mm, off)
    return keys


def main():
    INT_INDEX = '/home/xingrui/bench/rabitq_refactor/ivf/cohere_1m_ivf_rabitq_norm1.index'
    DED_INDEX = '/home/xingrui/bench/rabitq_refactor/ivf_dedicated/cohere_1m_ivfrabitq_dedicated.index'

    print('=== integrated:', INT_INDEX)
    c = Container(INT_INDEX)
    print(f'magic={c.magic:#x} segments ({len(c.segments)}):')
    for name, (off, sz, pad) in sorted(c.segments.items()):
        print(f'  {name:32s} off={off:>12} size={sz:>12} pad={pad}')
    total, list_cnt, block_vecs, block_cnt, lists = load_int_list_meta(c)
    print(f'header: total={total} lists={list_cnt} block_vecs={block_vecs} '
          f'blocks={block_cnt}')
    vec_sum = sum(v for _, _, v, _ in lists)
    print(f'sum(vector_count)={vec_sum}')
    int_keys = build_key_map(c, 'hc.keys', vec_sum)
    key2list = {}
    dup = 0
    id_base = 0
    for l, (_, _, vc, _) in enumerate(lists):
        for r in range(vc):
            k = int_keys[id_base + r]
            if k in key2list:
                dup += 1
            key2list[k] = (l, r)
        id_base += vc
    print(f'int key map: {len(key2list)} unique keys, dups={dup}')
    c.close()

    print('\n=== dedicated:', DED_INDEX)
    c2 = Container(DED_INDEX)
    print(f'magic={c2.magic:#x} segments ({len(c2.segments)}):')
    for name, (off, sz, pad) in sorted(c2.segments.items()):
        print(f'  {name:32s} off={off:>12} size={sz:>12} pad={pad}')
    (total, cluster_cnt, dim, padded, ex_bits, metric,
     clusters) = load_ded_cluster_meta(c2)
    print(f'header: total={total} clusters={cluster_cnt} dim={dim} '
          f'padded={padded} ex_bits={ex_bits} metric={metric}')
    vec_sum2 = sum(v for _, _, v, _, _ in clusters)
    print(f'sum(vector_count)={vec_sum2}')
    ded_keys = build_key_map(c2, 'ivf_rabitq.keys', vec_sum2)
    key2cluster = {}
    dup2 = 0
    for cl, (_, _, vc, _, k_off) in enumerate(clusters):
        for r in range(vc):
            k = ded_keys[k_off + r]
            if k in key2cluster:
                dup2 += 1
            key2cluster[k] = (cl, r)
    print(f'ded key map: {len(key2cluster)} unique keys, dups={dup2}')
    c2.close()

    inter = len(set(key2list) & set(key2cluster))
    print(f'\nkey set intersection: {inter}')

    print('\n=== nested ivf.centroid container (integrated) ===')
    coff, csz, _ = c.segments.get('ivf.centroid', (0, 0, 0))
    cc = Container(INT_INDEX, base=coff, size=csz)
    print(f'magic={cc.magic:#x} segments ({len(cc.segments)}):')
    for name, (off, sz, pad) in sorted(cc.segments.items()):
        print(f'  {name:32s} off={off:>12} size={sz:>12} pad={pad}')
    fo, fs, _ = cc.segments['flat.features']
    import math
    norms = []
    extras = []
    for r in range(8):
        row = struct.unpack_from('<768f', cc.mm, fo + r * 3076)
        extra = struct.unpack_from('<f', cc.mm, fo + r * 3076 + 3072)
        norms.append(round(math.sqrt(sum(x * x for x in row)), 4))
        extras.append(round(extra[0], 6))
    print(f'  stride=3076 rows0-7 norms: {norms}')
    print(f'  stride=3076 rows0-7 extra float: {extras}')
    last_row = struct.unpack_from('<768f', cc.mm, fo + 1023 * 3076)
    print(f'  row1023 norm: {math.sqrt(sum(x*x for x in last_row)):.4f}')
    tail = struct.unpack_from('<16f', cc.mm, fo + 1024 * 3072)
    print(f'  tail-as-3072 first16 floats: {tail}')
    ko, ks, _ = cc.segments['flat.keys']
    print('  flat.keys first8:', struct.unpack_from('<8Q', cc.mm, ko))
    imo, ims, _ = cc.segments['IndexMeta']
    print('  IndexMeta raw:', cc.mm[imo:imo + min(ims, 600)])

    # Empirical: is flat.features the ORIGINAL-space centroid matrix?
    # For sampled vectors, argmax_c cos(v, centroid_c) must equal the list
    # assigned by the index key map.
    VECS = ('/home/xingrui/bench/dataset/cohere/1m/'
            'cohere_train_vector_1m.new.centaur.vecs')
    vf = open(VECS, 'rb')
    vmm = mmap.mmap(vf.fileno(), 0, access=mmap.ACCESS_READ)
    nrows = 1000000
    vkeys = struct.unpack_from(f'<{nrows}Q', vmm, 4295 + nrows * 3072)
    vrow = {k: i for i, k in enumerate(vkeys)}
    cents = []
    for r in range(1024):
        cents.append(struct.unpack_from('<768f', cc.mm, fo + r * 3076))
    import random
    random.seed(7)
    sample = random.sample(list(key2list), 200)
    match = 0
    mism = []
    for k in sample:
        row = vrow[k]
        vec = struct.unpack_from('<768f', vmm, 4295 + row * 3072)
        best_c, best_s = -1, -2.0
        for ci, cvec in enumerate(cents):
            s = sum(a * b for a, b in zip(vec, cvec))
            if s > best_s:
                best_s, best_c = s, ci
        if best_c == key2list[k][0]:
            match += 1
        else:
            mism.append((k, key2list[k][0], best_c, round(best_s, 4)))
    print(f'  argmax-centroid==assigned-list: {match}/200')
    if mism:
        print(f'  mismatches (key, assigned, argmax, sim): {mism[:5]}')
    vmm.close()
    vf.close()
    cc.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
