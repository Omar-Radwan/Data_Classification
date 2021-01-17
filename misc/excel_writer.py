import os

import xlwt
from xlrd import open_workbook
from xlwt import Workbook
from xlutils.copy import copy
from misc.constants import *


class ExcelWriter:
    def __init__(self, sheet_name):
        self.sheet_name = sheet_name
        if not os.path.exists(WORK_BOOK_PATH):
            wb = Workbook()
            wb.add_sheet('dummy')
            wb.save(WORK_BOOK_PATH)

    def edit_sheet(self, report_dict=None):
        rb = open_workbook(WORK_BOOK_PATH)
        wb = copy(rb)

        s = wb.add_sheet(self.sheet_name)

        style = xlwt.XFStyle()

        s.write(0, 0, 'Average', style=style)
        s.write(1, 0, MICRO, style=style)
        s.write(2, 0, WEIGHTED, style=style)
        s.write(3, 0, MACRO, style=style)

        s.write(0, 1, F1_SCORE, style=style)
        s.write(1, 1, report_dict[F1_SCORE][MICRO])
        s.write(2, 1, report_dict[F1_SCORE][WEIGHTED])
        s.write(3, 1, report_dict[F1_SCORE][MACRO])

        s.write(0, 2, RECALL_SCORE, style=style)
        s.write(1, 2, report_dict[RECALL_SCORE][MICRO])
        s.write(2, 2, report_dict[RECALL_SCORE][WEIGHTED])
        s.write(3, 2, report_dict[RECALL_SCORE][MACRO])

        s.write(0, 3, PRECISION_SCORE, style=style)
        s.write(1, 3, report_dict[PRECISION_SCORE][MICRO])
        s.write(2, 3, report_dict[PRECISION_SCORE][WEIGHTED])
        s.write(3, 3, report_dict[PRECISION_SCORE][MACRO])

        s.write(5, 0, SCORE, style=style)
        s.write(6, 0, report_dict[SCORE])

        s.write(8, 1, "Actually g", style=style)
        s.write(8, 2, "Actually h", style=style)
        s.write(8, 3, "Total", style=style)

        s.write(9, 0, "Predicted g", style=style)
        s.write(10, 0, "Predicted h", style=style)
        s.write(11, 0, "Total", style=style)

        s.write(9, 1, int(report_dict[CONFUSION_MATRIX][0][0]))
        s.write(9, 2, int(report_dict[CONFUSION_MATRIX][0][1]))
        s.write(9, 3, int(report_dict[CONFUSION_MATRIX][0][0] + report_dict[CONFUSION_MATRIX][0][1]), style=style)

        s.write(10, 1, int(report_dict[CONFUSION_MATRIX][1][0]))
        s.write(10, 2, int(report_dict[CONFUSION_MATRIX][1][1]))
        s.write(10, 3, int(report_dict[CONFUSION_MATRIX][1][0] + report_dict[CONFUSION_MATRIX][1][1]), style=style)

        s.write(11, 1, int(report_dict[CONFUSION_MATRIX][0][0] + report_dict[CONFUSION_MATRIX][1][0]), style=style)
        s.write(11, 2, int(report_dict[CONFUSION_MATRIX][0][1] + report_dict[CONFUSION_MATRIX][1][1]), style=style)

        wb.save(WORK_BOOK_PATH)
