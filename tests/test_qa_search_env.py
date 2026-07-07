from common.rewards.search_policy import apply_no_search_policy


def test_no_search_policy_caps_positive_reward():
    assert (
        apply_no_search_policy(
            1.0,
            searched=False,
            require_search=True,
            no_search_penalty=-0.2,
        )
        == -0.2
    )


def test_no_search_policy_keeps_existing_negative_reward():
    assert (
        apply_no_search_policy(
            -0.5,
            searched=False,
            require_search=True,
            no_search_penalty=-0.2,
        )
        == -0.5
    )


def test_no_search_policy_allows_reward_after_search():
    assert (
        apply_no_search_policy(
            1.0,
            searched=True,
            require_search=True,
            no_search_penalty=-0.2,
        )
        == 1.0
    )
