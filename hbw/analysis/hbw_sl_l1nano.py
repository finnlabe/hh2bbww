# coding: utf-8

"""
Main analysis object for the nonresonant HH -> bbWW(SL) analysis
"""

from hbw.analysis.create_analysis import create_hbw_analysis

hbw_sl_l1nano = create_hbw_analysis(
    "hbw_sl_l1nano", 3,
    tags={
        "is_sl",
        "is_nonresonant",
        "is_l1nano",
        # "custom_signals",
    },
)
