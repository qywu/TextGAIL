#include <torch/extension.h>
#include <iostream>
#include <vector>

torch::Tensor d_pad_sequence(std::vector<torch::Tensor> tensors, bool batch_first, float padding_value)
{
    // assuming trailing dimensions and type of all the Tensors
    // in sequences are same and fetching those from tensors[0]
    if (tensors.size() == 0)
    {
        AT_ERROR("pad_sequence: input tensor sequence must have at least one tensor.");
    }
    auto max_size = tensors[0].sizes();
    std::vector<int64_t> output_size(max_size.size() + 1);
    std::copy(max_size.begin() + 1, max_size.end(), output_size.begin() + 2);

    // Find the max sequence length
    int64_t max_len = 0;
    for (auto &t : tensors)
    {
        auto size = t.size(0);
        if (size > max_len)
        {
            max_len = size;
        }
    }

    if (batch_first)
    {
        output_size[0] = tensors.size();
        output_size[1] = max_len;
    }
    else
    {
        output_size[0] = max_len;
        output_size[1] = tensors.size();
    }

    auto result = tensors[0].new_empty(output_size).fill_(padding_value);

    for (auto it = tensors.begin(); it != tensors.end(); ++it)
    {
        auto i = it - tensors.begin();
        auto seq_len = it->size(0);

        torch::Tensor view;

        if (batch_first)
        {
            // result[i, :seq_len, ...]
            view = result.narrow(0, i, 1).narrow(1, 0, seq_len).squeeze(0);
        }
        else
        {
            // result[:seq_len, i, ...]
            view = result.narrow(0, 0, seq_len).narrow(1, i, 1).squeeze(1);
        }
        view.copy_(*it);
    }
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("pad_sequence", &d_pad_sequence, "pad sequence for varied length");
}
