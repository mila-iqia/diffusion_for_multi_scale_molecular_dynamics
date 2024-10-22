import pytest
import torch
from crystal_diffusion.models.egnn_utils import (unsorted_segment_mean,
                                                 unsorted_segment_sum)


@pytest.fixture()
def num_messages():
    return 15


@pytest.fixture()
def num_ids():
    return 3


@pytest.fixture()
def message_ids(num_messages, num_ids):
    return torch.randint(low=0, high=num_ids, size=(num_messages,))


@pytest.fixture()
def num_message_features():
    return 2


@pytest.fixture()
def messages(num_messages, num_message_features):
    return torch.randn(num_messages, num_message_features)


def test_unsorted_segment_sum(num_messages, num_ids, message_ids, num_message_features, messages):
    expected_message_sums = torch.zeros(num_ids, num_message_features)
    for i in range(num_messages):
        m_id = message_ids[i]
        message = messages[i]
        expected_message_sums[m_id] += message

    message_summed = unsorted_segment_sum(messages, message_ids, num_ids)
    assert message_summed.size() == torch.Size((num_ids, num_message_features))
    assert torch.allclose(message_summed, expected_message_sums)


def test_unsorted_segment_mean(num_messages, num_ids, message_ids, num_message_features, messages):
    expected_message_sums = torch.zeros(num_ids, num_message_features)
    expected_counts = torch.zeros(num_ids, 1)
    for i in range(num_messages):
        m_id = message_ids[i]
        message = messages[i]
        expected_message_sums[m_id] += message
        expected_counts[m_id] += 1
    expected_message_average = expected_message_sums / torch.maximum(expected_counts, torch.ones_like(expected_counts))

    message_averaged = unsorted_segment_mean(messages, message_ids, num_ids)
    assert message_averaged.size() == torch.Size((num_ids, num_message_features))
    assert torch.allclose(message_averaged, expected_message_average)
