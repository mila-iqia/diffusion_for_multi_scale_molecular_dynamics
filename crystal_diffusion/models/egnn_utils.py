import torch


def unsorted_segment_sum(data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    """Sum all the elements in data by their ids.

    For example, data could be messages from atoms j to i. We want to sum all messages going to i, i.e. sum all elements
    in the message tensor that are going to i. This is indicated by the segment_ids input.

    Args:
        data: tensor to aggregate. Size is
            (number of elements to aggregate (e.g. number of edges in the message example), number of features)
        segment_ids: ids of each element in data (e.g. messages going to node i in the message example)
        num_segments: number of distinct elements in the data tensor

    Returns:
        tensor with the sum of data elements over ids. size: (num_segments, number of features)
    """
    result_shape = (num_segments, data.size(1))  # output size
    result = torch.zeros(result_shape).to(data)  # results starting as zeros - same dtype and device as data
    # tensor size manipulation to use a scatter_add operation
    # from (number of elements) to (number of elements, number of features) i.e. same size as data
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    # segment_ids needs to have the same size as data for the backward pass to go through
    # see https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_add_.html#torch.Tensor.scatter_add_
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data: torch.Tensor, segment_ids: torch.Tensor, num_segments: int) -> torch.Tensor:
    """Average all the elements in data by their ids.

    For example, data could be messages from atoms j to i. We want to average all messages going to i
    i.e. average all elements in the message tensor that are going to i. This is indicated by the segment_ids input.

    Args:
        data: tensor to aggregate. Size is
            (number of elements to aggregate (e.g. number of edges in the message example), number of features)
        segment_ids: ids of each element in data (e.g. messages going to node i in the message example)
        num_segments: number of distinct elements in the data tensor

    Returns:
        tensor with the average of data elements over ids. size: (num_segments, number of features)
    """
    result_shape = (num_segments, data.size(1))  # output size
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))  # tensor size manipulation for the backward pass
    result = torch.zeros(result_shape).to(data)  # sum the component
    count = torch.zeros(result_shape).to(data)  # count the number of data elements for each id to take the average
    result.scatter_add_(0, segment_ids, data)  # sum the data elements
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)  # avoid dividing by zeros by clamping the counts to be at least 1
