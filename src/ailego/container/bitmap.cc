// Copyright 2025-present the zvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "bitmap.h"

namespace zvec {
namespace ailego {

size_t Bitset::BitwiseAndCardinality(const Bitset &lhs, const Bitset &rhs) {
  return BitsetHelper::BitwiseAndCardinality(
      lhs.array_.data(), rhs.array_.data(),
      std::min(lhs.array_.size(), rhs.array_.size()));
}

size_t Bitset::BitwiseAndnotCardinality(const Bitset &lhs, const Bitset &rhs) {
  size_t lsize = lhs.array_.size();
  size_t rsize = rhs.array_.size();

  if (lsize > rsize) {
    return (
        BitsetHelper::BitwiseAndnotCardinality(lhs.array_.data(),
                                               rhs.array_.data(), rsize) +
        BitsetHelper::Cardinality(lhs.array_.data() + rsize, lsize - rsize));
  }
  return BitsetHelper::BitwiseAndnotCardinality(lhs.array_.data(),
                                                rhs.array_.data(), lsize);
}

size_t Bitset::BitwiseXorCardinality(const Bitset &lhs, const Bitset &rhs) {
  size_t lsize = lhs.array_.size();
  size_t rsize = rhs.array_.size();

  if (lsize < rsize) {
    return (
        BitsetHelper::BitwiseXorCardinality(lhs.array_.data(),
                                            rhs.array_.data(), lsize) +
        BitsetHelper::Cardinality(rhs.array_.data() + lsize, rsize - lsize));
  } else if (lsize > rsize) {
    return (
        BitsetHelper::BitwiseXorCardinality(lhs.array_.data(),
                                            rhs.array_.data(), rsize) +
        BitsetHelper::Cardinality(lhs.array_.data() + rsize, lsize - rsize));
  }
  return BitsetHelper::BitwiseXorCardinality(lhs.array_.data(),
                                             rhs.array_.data(), lsize);
}

size_t Bitset::BitwiseOrCardinality(const Bitset &lhs, const Bitset &rhs) {
  size_t lsize = lhs.array_.size();
  size_t rsize = rhs.array_.size();

  if (lsize < rsize) {
    return (
        BitsetHelper::BitwiseOrCardinality(lhs.array_.data(), rhs.array_.data(),
                                           lsize) +
        BitsetHelper::Cardinality(rhs.array_.data() + lsize, rsize - lsize));
  } else if (lsize > rsize) {
    return (
        BitsetHelper::BitwiseOrCardinality(lhs.array_.data(), rhs.array_.data(),
                                           rsize) +
        BitsetHelper::Cardinality(lhs.array_.data() + rsize, lsize - rsize));
  }
  return BitsetHelper::BitwiseOrCardinality(lhs.array_.data(),
                                            rhs.array_.data(), lsize);
}

void Bitmap::clear(void) {
  for (std::vector<Bucket *>::iterator iter = array_.begin();
       iter != array_.end(); ++iter) {
    delete (*iter);
  }
  array_.clear();
}

void Bitmap::copy(const Bitmap &rhs) {
  this->clear();

  for (std::vector<Bucket *>::const_iterator iter = rhs.array_.begin();
       iter != rhs.array_.end(); ++iter) {
    Bucket *bucket = nullptr;
    if (*iter) {
      bucket = new Bucket(*(*iter));
    }
    array_.push_back(bucket);
  }
}

void Bitmap::shrink_to_fit(void) {
  size_t shrink_count = 0;
  std::vector<Bucket *>::reverse_iterator iter;

  for (iter = array_.rbegin(); iter != array_.rend(); ++iter) {
    if (*iter) {
      if (!(*iter)->test_none()) {
        break;
      }
      delete (*iter);
      *iter = nullptr;
    }
    ++shrink_count;
  }
  for (; iter != array_.rend(); ++iter) {
    if ((*iter) && (*iter)->test_none()) {
      delete (*iter);
      *iter = nullptr;
    }
  }
  if (shrink_count != 0) {
    array_.resize(array_.size() - shrink_count);
  }
}

bool Bitmap::test(size_t num) const {
  // High 16 bits
  size_t offset = num >> 16;

  if (offset < array_.size()) {
    const Bucket *bucket = array_[offset];
    if (bucket) {
      // Low 16 bits
      return bucket->test(static_cast<uint16_t>(num));
    }
  }
  return false;
}

void Bitmap::set(size_t num) {
  // High 16 bits
  size_t offset = num >> 16;
  if (offset >= array_.size()) {
    array_.resize(offset + 1, nullptr);
  }

  Bucket *&bucket = array_[offset];
  if (!bucket) {
    bucket = new Bucket;
  }
  // Low 16 bits
  bucket->set(static_cast<uint16_t>(num));
}

void Bitmap::reset(size_t num) {
  // High 16 bits
  size_t offset = num >> 16;
  if (offset >= array_.size()) {
    array_.resize(offset + 1, nullptr);
  }

  if (offset < array_.size()) {
    Bucket *bucket = array_[offset];
    if (bucket) {
      // Low 16 bits
      bucket->reset(static_cast<uint16_t>(num));
    }
  }
}

void Bitmap::flip(size_t num) {
  // High 16 bits
  uint16_t offset = (uint16_t)(num >> 16);
  if (offset >= array_.size()) {
    array_.resize(offset + 1, nullptr);
  }

  Bucket *&bucket = array_[offset];
  if (!bucket) {
    bucket = new Bucket;
  }
  // Low 16 bits
  bucket->flip(static_cast<uint16_t>(num));
}

void Bitmap::bitwise_and(const Bitmap &rhs) {
  size_t overlap = std::min(array_.size(), rhs.array_.size());

  for (size_t i = 0; i < overlap; ++i) {
    Bucket *&dst = array_[i];

    if (dst) {
      const Bucket *src = rhs.array_[i];
      if (src) {
        dst->bitwise_and(*src);
      } else {
        delete dst;
        dst = nullptr;
      }
    }
  }
  for (size_t i = overlap; i < array_.size(); ++i) {
    Bucket *&dst = array_[i];
    delete dst;
    dst = nullptr;
  }
}

void Bitmap::bitwise_andnot(const Bitmap &rhs) {
  size_t overlap = std::min(array_.size(), rhs.array_.size());

  for (size_t i = 0; i < overlap; ++i) {
    Bucket *&dst = array_[i];

    if (dst) {
      const Bucket *src = rhs.array_[i];
      if (src) {
        dst->bitwise_andnot(*src);
      }
    }
  }
}

void Bitmap::bitwise_or(const Bitmap &rhs) {
  size_t overlap = std::min(array_.size(), rhs.array_.size());

  for (size_t i = 0; i < overlap; ++i) {
    const Bucket *src = rhs.array_[i];

    if (src) {
      Bucket *&dst = array_[i];

      if (dst) {
        dst->bitwise_or(*src);
      } else {
        dst = new Bucket(*src);
      }
    }
  }
  for (size_t i = overlap; i < rhs.array_.size(); ++i) {
    const Bucket *src = rhs.array_[i];
    Bucket *bucket = nullptr;

    if (src) {
      bucket = new Bucket(*src);
    }
    array_.push_back(bucket);
  }
}

void Bitmap::bitwise_xor(const Bitmap &rhs) {
  size_t overlap = std::min(array_.size(), rhs.array_.size());

  for (size_t i = 0; i < overlap; ++i) {
    const Bucket *src = rhs.array_[i];

    if (src) {
      Bucket *&dst = array_[i];

      if (dst) {
        dst->bitwise_xor(*src);
      } else {
        dst = new Bucket(*src);
      }
    }
  }
  for (size_t i = overlap; i < rhs.array_.size(); ++i) {
    const Bucket *src = rhs.array_[i];
    Bucket *bucket = nullptr;

    if (src) {
      bucket = new Bucket(*src);
    }
    array_.push_back(bucket);
  }
}

void Bitmap::bitwise_not(void) {
  for (std::vector<Bucket *>::iterator iter = array_.begin();
       iter != array_.end(); ++iter) {
    Bucket *&bucket = *iter;
    if (!bucket) {
      bucket = new Bucket;
    }
    bucket->bitwise_not();
  }
}

bool Bitmap::test_all(void) const {
  if (array_.empty()) {
    return false;
  }
  for (std::vector<Bucket *>::const_iterator iter = array_.begin();
       iter != array_.end(); ++iter) {
    if (!(*iter) || !(*iter)->test_all()) {
      return false;
    }
  }
  return true;
}

bool Bitmap::test_any(void) const {
  for (std::vector<Bucket *>::const_iterator iter = array_.begin();
       iter != array_.end(); ++iter) {
    if (*iter && (*iter)->test_any()) {
      return true;
    }
  }
  return false;
}

bool Bitmap::test_none(void) const {
  for (std::vector<Bucket *>::const_iterator iter = array_.begin();
       iter != array_.end(); ++iter) {
    if (*iter && !(*iter)->test_none()) {
      return false;
    }
  }
  return true;
}

size_t Bitmap::cardinality(void) const {
  size_t result = 0;
  for (std::vector<Bucket *>::const_iterator iter = array_.begin();
       iter != array_.end(); ++iter) {
    if (*iter) {
      result += (*iter)->cardinality();
    }
  }
  return result;
}

void Bitmap::extract(size_t base, std::vector<size_t> *out) const {
  for (std::vector<Bucket *>::const_iterator iter = array_.begin();
       iter != array_.end(); ++iter) {
    if (*iter) {
      (*iter)->extract(base, out);
    }
    base += Bucket::MAX_SIZE;
  }
}

size_t Bitmap::BitwiseAndCardinality(const Bitmap &lhs, const Bitmap &rhs) {
  size_t overlap = std::min(lhs.array_.size(), rhs.array_.size());
  size_t dist = 0;

  for (size_t i = 0; i < overlap; ++i) {
    const Bucket *l = lhs.array_[i];
    const Bucket *r = rhs.array_[i];

    if (l && r) {
      dist += Bucket::BitwiseAndCardinality(*l, *r);
    }
  }
  return dist;
}

size_t Bitmap::BitwiseAndnotCardinality(const Bitmap &lhs, const Bitmap &rhs) {
  size_t overlap = std::min(lhs.array_.size(), rhs.array_.size());
  size_t dist = 0;

  for (size_t i = 0; i < overlap; ++i) {
    const Bucket *l = lhs.array_[i];
    if (l) {
      const Bucket *r = rhs.array_[i];
      if (r) {
        dist += Bucket::BitwiseAndnotCardinality(*l, *r);
      } else {
        dist += l->cardinality();
      }
    }
  }
  for (size_t i = overlap; i < lhs.array_.size(); ++i) {
    const Bucket *l = lhs.array_[i];
    if (l) {
      dist += l->cardinality();
    }
  }
  return dist;
}

size_t Bitmap::BitwiseXorCardinality(const Bitmap &lhs, const Bitmap &rhs) {
  size_t overlap = std::min(lhs.array_.size(), rhs.array_.size());
  size_t dist = 0;

  for (size_t i = 0; i < overlap; ++i) {
    const Bucket *l = lhs.array_[i];
    const Bucket *r = rhs.array_[i];

    if (l && r) {
      dist += Bucket::BitwiseXorCardinality(*l, *r);
    } else if (l) {
      dist += l->cardinality();
    } else if (r) {
      dist += r->cardinality();
    }
  }
  for (size_t i = overlap; i < lhs.array_.size(); ++i) {
    const Bucket *l = lhs.array_[i];
    if (l) {
      dist += l->cardinality();
    }
  }
  for (size_t i = overlap; i < rhs.array_.size(); ++i) {
    const Bucket *r = rhs.array_[i];
    if (r) {
      dist += r->cardinality();
    }
  }
  return dist;
}

size_t Bitmap::BitwiseOrCardinality(const Bitmap &lhs, const Bitmap &rhs) {
  size_t overlap = std::min(lhs.array_.size(), rhs.array_.size());
  size_t dist = 0;

  for (size_t i = 0; i < overlap; ++i) {
    const Bucket *l = lhs.array_[i];
    const Bucket *r = rhs.array_[i];

    if (l && r) {
      dist += Bucket::BitwiseOrCardinality(*l, *r);
    } else if (l) {
      dist += l->cardinality();
    } else if (r) {
      dist += r->cardinality();
    }
  }
  for (size_t i = overlap; i < lhs.array_.size(); ++i) {
    const Bucket *l = lhs.array_[i];
    if (l) {
      dist += l->cardinality();
    }
  }
  for (size_t i = overlap; i < rhs.array_.size(); ++i) {
    const Bucket *r = rhs.array_[i];
    if (r) {
      dist += r->cardinality();
    }
  }
  return dist;
}

}  // namespace ailego
}  // namespace zvec