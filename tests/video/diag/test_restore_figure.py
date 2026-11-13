"""The layout of the stacked restore pictures (diagnosis, M17)."""

from exp.video.diag.restore_figure import LABEL_W, TITLE_H, layout


def test_layout_stacks_the_pictures_under_their_titles():
    size, corners = layout([(100, 60), (140, 30)])
    assert size == (LABEL_W + 140, TITLE_H + 60 + TITLE_H + 30)
    assert corners == [(LABEL_W, TITLE_H), (LABEL_W, 2 * TITLE_H + 60)]
