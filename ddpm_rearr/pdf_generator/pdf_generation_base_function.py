import copy

from reportlab.platypus import SimpleDocTemplate, Spacer, PageBreak
from reportlab.platypus import Paragraph, Image, Table
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import reportlab.lib.colors as colors


class PDFElements:
    """
    In this class,
    """
    def __init__(self, pdf_file_name):
        self.story = []
        stylesheet = getSampleStyleSheet()
        self.title_style = stylesheet['Title']
        self.heading_style = stylesheet['Heading2']
        self.normal_style = stylesheet['Normal']
        self.bold_style = copy.deepcopy(stylesheet['Normal'])
        self.bold_style.fontName = '%s-Bold' % self.bold_style.fontName
        self.doc = SimpleDocTemplate(pdf_file_name, pagesize=A4,
                                     leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                                     topMargin=1.5 * cm, bottomMargin=2.5 * cm)

    def add_page_break(self):
        self.story.append(PageBreak())

    def add_one_blank_line(self):
        self.story.append(Spacer(1, 0.5 * cm))

    def add_title(self, title: str):
        """
        add title into current page
        :param title:
        :return:
        """
        self.story.append(Paragraph(title, self.title_style))

    def add_heading(self, heading_string: str):
        """
        add heading into current page
        :param heading_string:
        :return:
        """
        self.story.append(Paragraph(heading_string, self.heading_style))

    def add_paragraph(self, paragraph_string: str):
        """
        adding one paragraph into current page
        :param paragraph_string: paragraph string
        :return:
        """
        style = self.normal_style
        self.add_one_blank_line()
        self.story.append(Paragraph(paragraph_string, style))

    def add_bold_paragraph(self, paragraph_string: str):
        """
        adding one bold paragraph into current page
        :param paragraph_string:
        :return:
        """
        style = self.bold_style
        self.add_one_blank_line()
        self.story.append(Paragraph(paragraph_string, style))

    def add_table(self, table_title: str, table_list: list):
        """
        adding table into current page
        :param table title
        :param table_list:
        :return:
        """
        style = self.normal_style
        self.add_one_blank_line()
        self.story.append(Paragraph(table_title, style))
        self.add_one_blank_line()
        self.story.append(Table(table_list, hAlign='LEFT', style=[('GRID',(0,0),(-1,-1),0.5,colors.black)]))

    def add_one_image(self, image_path, image_show_size_x, image_show_size_y, image_title):
        """
        adding one image into the current page
        :param image_path:
        :param image_show_size_x:
        :param image_show_size_y:
        :return:
        """
        style = self.normal_style

        self.add_one_blank_line()
        self.story.append(Paragraph(image_title, style))

        im1 = Image(image_path, width=image_show_size_x, height=image_show_size_y)
        summary_table = [[im1]]
        summary_table_style = []
        self.story.append(Table(summary_table, style=summary_table_style))

    def add_two_images_per_column(self, image_path1, image_path2, image_show_size_x, image_show_size_y, image_title):
        """
        put two images in the current page in one column, assume each image has the same image to show size
        :param image_path1:
        :param image_path2:
        :param image_show_size_x:
        :param image_show_size_y:
        :return:
        """

        style = self.normal_style

        self.add_one_blank_line()
        self.story.append(Paragraph(image_title, style))

        self.add_one_blank_line()
        im1 = Image(image_path1, width=image_show_size_x, height=image_show_size_y)
        im2 = Image(image_path2, width=image_show_size_x, height=image_show_size_y)

        summary_table = [[im1, im2]]
        summary_table_style = []
        self.story.append(Table(summary_table, style=summary_table_style))

    def add_three_images_per_column(self, image_path1, image_path2, image_path3, image_show_size_x, image_show_size_y, image_title):
        """
        put three images in the current page in one column, assume each image has the same image to show size
        :param image_path1:
        :param image_path2:
        :param image_path2:
        :param image_show_size_x:
        :param image_show_size_y:
        :return:
        """

        style = self.normal_style

        self.add_one_blank_line()
        self.story.append(Paragraph(image_title, style))

        self.add_one_blank_line()
        im1 = Image(image_path1, width=image_show_size_x, height=image_show_size_y)
        im2 = Image(image_path2, width=image_show_size_x, height=image_show_size_y)
        im3 = Image(image_path3, width=image_show_size_x, height=image_show_size_y)

        summary_table = [[im1, im2, im3]]
        summary_table_style = []
        self.story.append(Table(summary_table, style=summary_table_style))

    def build(self):
        self.doc.build(self.story)


def function_demo():
    """
    class tests
    :return:
    """

    report = PDFElements('sample.pdf')
    title_str = 'Title test'
    report.add_title(title_str)
    heading_str = 'heading_test'
    report.add_title(heading_str)

    report.add_paragraph('normal text test')
    for i in range(10):
        paragraph_str = str(i)*(i+1)
        report.add_paragraph(paragraph_str)
    report.add_page_break()

    report.add_bold_paragraph('bold text test')
    for i in range(10):
        paragraph_str = str(i) * (i + 1)
        report.add_bold_paragraph(paragraph_str)
    report.add_page_break()

    heading_str = 'table_test'
    report.add_title(heading_str)
    table_str = []
    for i in range(10):
        table_str.append([i, i+1, i+2])
    table_title = 'table 1'
    report.add_table(table_title, table_str)
    report.add_page_break()

    heading_str = 'image_test'
    report.add_title(heading_str)
    img_path = '/Users/strdrv/Desktop/2.16.840.114490.1.3.3883751178.6988.1614710990.567.png'
    image_title = 'image 1'
    image_to_show_resolution = 600
    report.add_one_image(img_path, image_to_show_resolution, image_to_show_resolution, image_title)
    report.add_page_break()

    heading_str = '2 images test'
    report.add_title(heading_str)
    img_path1 = '/Users/strdrv/Desktop/2.16.840.114490.1.3.3883751178.6988.1614710990.567.png'
    img_path2 = '/Users/strdrv/Desktop/2.16.840.114490.1.3.3883751178.6988.1614710916.565.png'
    image_title = 'image 1 & 2'
    image_to_show_resolution = 300
    report.add_two_images_per_column(img_path1, img_path2, image_to_show_resolution, image_to_show_resolution, image_title)
    report.add_page_break()

    report.build()


if __name__ == '__main__':
    function_demo()
