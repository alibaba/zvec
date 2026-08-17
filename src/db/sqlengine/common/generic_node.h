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

#pragma once

#include <memory>
#include <string>

namespace zvec::sqlengine {

template <typename NodeOp, typename Node>
class GenericNode {
 public:
  using Ptr = std::shared_ptr<Node>;

  GenericNode(NodeOp m_op);
  virtual ~GenericNode() = default;

  void set_left(Ptr m_left);
  void set_right(Ptr m_right);
  const Ptr &left() const {
    return left_;
  }
  const Ptr &right() const {
    return right_;
  }
  Node *left_node() const {
    return left_.get();
  }
  Node *right_node() const {
    return right_.get();
  }
  void set_parent(GenericNode *m_parent);
  GenericNode *parent();

  virtual NodeOp op() const {
    return op_;
  }
  virtual void set_op(NodeOp value) {
    op_ = value;
  }
  virtual std::string text() const = 0;

 protected:
  NodeOp op_;
  Ptr left_{nullptr};
  Ptr right_{nullptr};
  GenericNode *parent_{nullptr};
};

template <typename NodeOp, typename Node>
GenericNode<NodeOp, Node>::GenericNode(NodeOp m_op) {
  op_ = m_op;
}

template <typename NodeOp, typename Node>
void GenericNode<NodeOp, Node>::set_left(Ptr m_left) {
  left_ = std::move(m_left);
  if (left_ != nullptr) {
    left_->set_parent(this);
  }
}
template <typename NodeOp, typename Node>
void GenericNode<NodeOp, Node>::set_right(Ptr m_right) {
  right_ = std::move(m_right);
  if (right_ != nullptr) {
    right_->set_parent(this);
  }
}
template <typename NodeOp, typename Node>
void GenericNode<NodeOp, Node>::set_parent(GenericNode<NodeOp, Node> *value) {
  this->parent_ = value;
}
template <typename NodeOp, typename Node>
GenericNode<NodeOp, Node> *GenericNode<NodeOp, Node>::parent() {
  return parent_;
}

}  // namespace zvec::sqlengine
