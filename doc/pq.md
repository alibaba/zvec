## 在turbo中实现preprocessor

### 1. Preprocessor

```cpp
class Preprocessor {
 public:
  using Pointer = std::shared_ptr<Preprocessor>;

  virtual ~Preprocessor() = default;

  //! Input dimensionality accepted by apply().
  virtual int in_dim() const = 0;

  //! Output dimensionality produced by apply(). May differ from in_dim()
  //! (e.g. FHT pads to the next power of two).
  virtual int out_dim() const = 0;

  //! Forward transform: map an input vector to the preprocessed space.
  //! \p out must hold at least out_dim() elements.
  virtual void apply(const float *in, float *out) const = 0;

  //! Inverse transform: recover the original-space vector from a preprocessed
  //! one. \p out must hold at least in_dim() elements.
  virtual void apply_inverse(const float *in, float *out) const = 0;

  //! Fit the preprocessor from a contiguous batch of training data.
  //! \p data  pointer to the first element of the batch.
  //! \p num   number of vectors in the batch.
  //! \p stride byte offset between consecutive vectors (0 ⇒ packed).
  virtual void train(const void *data, size_t num, size_t stride) = 0;

  //! Serialize the preprocessor into a self-contained blob (header + payload).
  virtual int serialize(std::string *out) const = 0;

  //! Deserialize the preprocessor from a raw, possibly mmap-backed buffer.
  virtual int deserialize(const void *data, size_t len) = 0;
};
```
定位：属于/src/turbo/quantizer的一部分，定义在/src/turbo/quantizer/preprocessor.h

### 2. FhtRotator

```cpp
class FhtRotator : public Preprocessor {
 public:
  using Pointer = std::shared_ptr<FhtRotator>;

  //! Kind of rotator.
  virtual RotatorType type() const = 0;

  // in_dim(), out_dim(), apply(), apply_inverse(), train(), serialize(),
  // and deserialize() are inherited from Preprocessor.
}

//! Create an untrained rotator of the given type for in_dim dimensions.
FhtRotator::Pointer CreateFhtRotator(int in_dim);

//! Create and restore a rotator from a serialized blob (reads the type from
//! the embedded RotatorSerHeader). Returns nullptr on malformed input.
FhtRotator::Pointer CreateFhtRotatorFromBlob(const void *data, size_t len);
```

### 3. 数据保存

#### 保存格式
```cpp
struct RotatorSerHeader {
  uint32_t magic;         // kRotatorMagic
  uint16_t version;       // kRotatorSerVersion
  uint16_t rotator_type;  // RotatorType
  uint32_t in_dim;        // input dimensionality
  uint32_t out_dim;       // output dimensionality
  uint32_t payload_size;  // bytes following the header
  uint32_t reserved;      // 0, for future use / alignment
};
static_assert(sizeof(RotatorSerHeader)) == 24,
```
定义写在preprocessor.h中

在FHT中，
1. rotator_type设置为1，表示FHT旋转
2. in_dim和out_dim设置为相同，
4. 具体实现参考/root/code/zvec/中fht的实现

#### 保存位置
RotatorSerHeader放到当前已经存在的QuantizerSerHeader中，和他一起保存。具体来说：
1. 量化器调用serialize -> 其中调用preprocessor的serialize
2. serialize后的数据保存到Indexmeta中
3. 当前方案已经有了保存Indexmeta，这样就可以自然保存


### 4. 分发函数
在 turbo.h/.cc 中新增函数，将fht的分发功能，也放在其中
