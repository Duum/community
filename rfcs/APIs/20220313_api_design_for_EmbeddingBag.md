# paddle.nn.EmbeddingBag 设计文档

|API名称 |  EmbeddingBag | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | 杜渺 | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2022-03-13 | 
|版本号 | V1.0 | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop | 
|文件名 | 20220313_api_design_for_EmbeddingBag.md<br> | 


# 一、概述
## 1、相关背景
 EmbeddingBag 是 Embedding 的拓展，在功能上相当于 Embedding + 求和/求均值/求最大值的操作。相比直接组合，EmbeddingBag 会有更高的计算效率和更小的内存消耗。
 这个功能是参考的[Pytorch，](https://pytorch.org/docs/stable/generated/torch.nn.EmbeddingBag.html)TensorFlow没有这个功能的官方实现。


## 2、功能目标
此任务的目标是在 Paddle 框架中，新增 EmbeddingBag 和 embedding_bag API。调用路径为：paddle.nn.EmbeddingBag 和 paddle.nn.functional.embedding_bag。

## 3、意义
为paddle增加EmbeddingBag，对于很多NLP任务，做NLP的句子嵌入比单独调用embedding + sum等更有效率 。

# 二、飞桨现状
paddle目前没有EmbeddingBag 这个实现，可以根据上层api搭建，但是时间和空间性能会有影响。


# 三、业内方案调研
### PyTorch
EmbeddingBag的cpu的实现位于pytorch/aten/src/ATen/native/EmbeddingBag.cpp 这个文件中
##### 前向传播
```cpp 
void _embedding_bag_cpu_impl_out(Tensor& output, Tensor& offset2bag,
                            Tensor& bag_size, Tensor& max_indices,
                            const Tensor &weight, const Tensor &indices,
                            const Tensor &offsets, const int64_t mode,
                            const c10::optional<Tensor>& per_sample_weights,
                            bool include_last_offset, int64_t padding_idx) {
  if (mode == MODE_MEAN || mode == MODE_SUM) {
    AT_DISPATCH_FLOATING_TYPES(weight.scalar_type(), "embedding_bag_no_grad_cpu_out",
      [&indices, &offset2bag, &per_sample_weights, &weight, &output, &offsets, &include_last_offset, &mode, &bag_size, &padding_idx]() {
      AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "embedding_bag_no_grad_cpu_out",
        [&indices, &offset2bag, &per_sample_weights, &weight, &output, &offsets, &include_last_offset, &mode, &bag_size, &padding_idx]() {
        if (per_sample_weights.has_value() && per_sample_weights.value().defined()) {
          TORCH_INTERNAL_ASSERT(mode == MODE_SUM);
          index_select_scale_add<scalar_t, index_t>(
            indices, offset2bag, per_sample_weights.value(), weight, output, offsets, include_last_offset, bag_size, padding_idx);
        } else {
          index_select_add<scalar_t, index_t>(indices, offset2bag, weight, output, offsets, include_last_offset, bag_size, padding_idx);
        }
      });
    });
    apply_bag_size(mode, output, bag_size);
    if (mode == MODE_SUM) {
      // make bag_size output deterministic
      at::native::zero_(bag_size);
    }
     max_indices.copy_(bag_size);
  } else { // MODE_MAX
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      weight.scalar_type(), "embedding_bag_cpu_max_out", [&]() {
        embedding_bag_cpu_max_out<scalar_t>(
          max_indices, weight, indices, offset2bag, output, include_last_offset, bag_size, padding_idx);
      }
    );
  }
}

```

里面分了三种模式，分别为MEAN,SUM, MAX
```cpp
template <typename scalar_t>
void embedding_bag_cpu_max_out(
    Tensor& max_indices,
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offset2bag,
    const Tensor& output,
    bool include_last_offset,
    Tensor& bag_size,
    int64_t padding_idx) {
  int64_t numIndices = indices.numel();
  int64_t featureSize = weight.size(1);
  int64_t vocab_size = weight.size(0);
  AT_DISPATCH_INDEX_TYPES(indices.scalar_type(), "embedding_bag_cpu_max_out", [&] {
    auto* indices_data = indices.data_ptr<index_t>();
    auto* offset2bag_data = offset2bag.data_ptr<index_t>();

    auto* max_indices_data = max_indices.data_ptr<index_t>();
    auto max_indices_stride = max_indices.strides()[0];

    auto* weight_data = weight.data_ptr<scalar_t>();
    auto* output_data = output.data_ptr<scalar_t>();
    auto* bag_size_data = bag_size.data_ptr<index_t>();
    auto weight_stride0 = weight.strides()[0];
    auto weight_stride1 = weight.strides()[1];
    auto output_stride = output.strides()[0];
    int64_t numBags = bag_size.size(0);
    std::vector<bool> bag_empty(numBags, true);

    for (const auto i : c10::irange(numIndices)) {
      auto bag = offset2bag_data[i];
      auto word_idx = indices_data[i];
      TORCH_CHECK(
          word_idx >= 0 && word_idx < vocab_size,
          "embedding_bag: Expected idx >= 0 && idx < num_embeddings but found idx to be ",
          word_idx);
      if (word_idx != static_cast<index_t>(padding_idx)) {
        bool is_first_for_bag = bag_empty[bag];
        for (const auto dim : c10::irange(featureSize)) {
          auto& current_item = output_data[output_stride * bag + dim];
          auto weight_item =
              weight_data[weight_stride0 * word_idx + dim * weight_stride1];

          if (is_first_for_bag || (weight_item > current_item)) {
            current_item = weight_item;
            max_indices_data[max_indices_stride * bag + dim] = word_idx;
          }
        }
        if (is_first_for_bag) {
          bag_empty[bag] = false;
        }
      } else {
        // Decrement bag_size to reflect that the index is padded
        bag_size_data[bag]--;
      }
    }
  });
}
```
MEAN

##### 反向传播

```cpp
Tensor& soft_margin_loss_backward_out(const Tensor& grad_output, const Tensor& input, const Tensor& target, int64_t reduction, Tensor& grad_input) {
  auto norm = reduction == Reduction::Mean ? 1. / input.numel() : 1.;
  auto z = at::exp(-target * input);
  // inplace version of: grad_input = -norm * target * z / (1. + z) * grad_output;
  at::mul_out(grad_input, target, z).mul_(-norm);
  z.add_(1);
  grad_input.div_(z).mul_(grad_output);
  return grad_input;
}
```

### tensorflow

tensorflow没有官方大算子实现，但是才Addon库中有实现。关于Addons的介绍
> TensorFlow Addons is a repository of contributions that conform to well-established API patterns, but implement new functionality not available in core TensorFlow. 

tensorflow中EmbeddingBag实现的功能相对于PyTorch比较简陋，没有offsets这个参数，只在能在T维度做固定的BAG输入维度为2，至3维度。
# 四、对比分析
两种方案对比：
- 内部实现SoftMarginLoss大算子的前向传播和反向传播，优点是直接操作矩阵计算流，性能最优；缺点是实现起来相对麻烦。
- 调用小算子拼接，优点是实现速度比较快，不用单独实现反向传播；缺点是有潜在的可能存在性能损失。

# 五、设计思路与实现方案
为paddle phi计算库内部添加SoftMarginLoss的前向传播和反向传播大算子（CPU和GPU各自单独实现）。然后为paddle 动态图和静态图分别添加SoftMarginLoss的API。
## 命名与参数设计
参考：[飞桨API 设计及命名规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_design_guidelines_standard_cn.html)
## 底层OP设计
添加到paddle最新维护的phi计算库中。
## API实现方案
新增两个api，调用路径为：
paddle.nn.SoftMarginLoss
和
paddle.nn.functional.soft
_margin_loss
# 六、测试和验收的考量
- Loss准确度的测试。
- 1D，2D tensor的表现行为和pytorch表现一致
参考：[新增API 测试及验收规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_accpetance_criteria_cn.html)

# 七、可行性分析和排期规划
- 3月20号合入rfcs
- 3月30号合入实现和测试，过ci


# 八、影响面
无

# 名词解释
无
# 附件及参考资料
https://pytorch.org/docs/stable/generated/torch.nn.SoftMarginLoss.html