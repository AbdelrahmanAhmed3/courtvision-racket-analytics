from courtvision.analytics.teams import FAR, NEAR, assign_teams


def test_each_player_belongs_to_the_half_they_play_in_most() -> None:
    positions = [
        *[(1, 0.8)] * 5,  # near half all rally
        *[(2, 0.2)] * 5,  # far half
        *[(3, 0.3)] * 4,
        (3, 0.7),  # far, except one frame chasing a ball near the net
    ]

    assert assign_teams(positions) == {1: NEAR, 2: FAR, 3: FAR}


def test_a_tie_goes_to_the_near_team() -> None:
    assert assign_teams([(5, 0.4), (5, 0.6)]) == {5: NEAR}
