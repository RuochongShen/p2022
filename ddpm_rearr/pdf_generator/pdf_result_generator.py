from datetime import datetime
import numpy as np

import sys
# sys.path.append('../../../../evaluation_utils/pdf_generation/')

from pdf_generation_base_function import PDFElements


def pdf_template():
    """

    :return:
    """
    report = PDFElements('resultFeb.pdf')
    currentDateAndTime = datetime.now()
    title_str = f'Results'
    report.add_title(title_str)
    heading_str = f'Time: {currentDateAndTime.year}/{currentDateAndTime.month}/{currentDateAndTime.day}'
    report.add_title(heading_str)

    report.add_title('newweight_sampling_ave')
    report.add_one_blank_line()
    report.add_paragraph("ave psnr: 6.6946; ave ssim: 0.2085; ave rmse: 2186.5447")
    report.add_one_blank_line()
    for i in range(20):
        string_to_input = f"scan {i}"
        DDPM_img_path = f"newweight_sampling_ave{i}/9.png"
        gt_img_path = f"newweight_sampling_ave{i}/gt.png"
        raw_img_path = f"newweight_sampling_ave{i}/raw.png"
        report.add_paragraph(string_to_input)
        report.add_three_images_per_column(DDPM_img_path, gt_img_path, raw_img_path, 150,
                                           150, 'DDPM average of 16, GT and raw')
        report.add_page_break()

    report.add_title('DDPM sino MAR')
    report.add_one_blank_line()
    report.add_paragraph("ave psnr:36.1883; ave ssim:0.9106; ave rmse:4.2401")
    report.add_one_blank_line()
    for i in range(200):
        string_to_input = f"scan {i}"
        DDPM_img_path = f"newweight_sampling_ave{i}/imDDPMMAR.png"
        gt_img_path = f"newweight_sampling_ave{i}/gt.png"
        raw_img_path = f"newweight_sampling_ave{i}/raw.png"
        report.add_paragraph(string_to_input)
        report.add_three_images_per_column(DDPM_img_path, gt_img_path, raw_img_path, 150,
                                           150, 'DDPM average MAR, GT and raw')
        report.add_page_break()

    report.add_title('DDPM Repaint')
    report.add_one_blank_line()
    report.add_paragraph("ave psnr: 28.8006; ave ssim: 0.8101; ave rmse: 66.575")
    report.add_one_blank_line()
    for i in range(20):
        string_to_input = f"scan {i}"
        DDPM_img_path = f"train_sampling{i}/9.png"
        gt_img_path = f"train_sampling{i}/gt.png"
        raw_img_path = f"train_sampling{i}/raw.png"
        report.add_paragraph(string_to_input)
        report.add_three_images_per_column(DDPM_img_path, gt_img_path, raw_img_path, 150,
                                           150, 'DDPM average MAR, GT and raw')
        report.add_page_break()

    heading_str = 'model performance table'
    report.add_title(heading_str)
    table_str = []
    table_str.append([f'model_name', f'PSNR', f'SSIM'])
    models = ['Input', 'LI', 'NMAR', 'CNNMAR', 'DuDoNet', 'DDPM', 'DDPMsinoMAR', 'DDPM Repaint']
    psnrs = [27.06, 29.27, 29.48, 30.38, 31.14, 6.6496, 36.1883, 28.8006]
    ssims = [0.7586, 0.9347, 0.9442, 0.9644, 0.9814, 0.2085, 0.9106, 0.8101]
    for i in range(8):
        table_str.append([f'{models[i]}', f'{psnrs[i]}', f'{ssims[i]}'])
    table_title = 'table 1: model performance table'
    report.add_table(table_title, table_str)
    report.add_page_break()

    report.build()


if __name__ == '__main__':
    pdf_template()
