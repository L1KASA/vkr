from docx.shared import Pt, Cm
from docx.enum.text import WD_LINE_SPACING, WD_ALIGN_PARAGRAPH
from docx import Document
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import flet as ft
from flet.matplotlib_chart import MatplotlibChart
import os

matplotlib.use("svg")
temp_filenames = ["temp_plot1.png", "temp_plot2.png"]

def main(page: ft.Page):
    def window_event(e):
        print(e.data)
        if e.data == "close":
            print('page closed')
        
            for filename in temp_filenames:
                print(filename)
                print(11111)
                if os.path.exists(filename):
                    print(12)
                    os.remove(filename) 
            page.window_destroy()

    page.window_prevent_close = True
    page.on_window_event = window_event
    page.scroll = "AUTO"   
    page.theme_mode = ft.ThemeMode.LIGHT
    
    def theme_changed(e):
        page.theme_mode = (
            ft.ThemeMode.DARK
            if page.theme_mode == ft.ThemeMode.LIGHT
            else ft.ThemeMode.LIGHT
        )
        c.label = (
            "Светлая тема" if page.theme_mode == ft.ThemeMode.LIGHT else "Темная тема"
        )
        page.update()

    page.theme_mode = ft.ThemeMode.LIGHT
    c = ft.Switch(label="Светлая тема", on_change=theme_changed)
    
    dlg = ft.AlertDialog(
        title=ft.Text(\
            "Тема: Исследование влияния природно-климатических факторов на исторические процессы методом математического моделирования\nПочта автора: tahmazova_anzhelika@mail.ru")\
            , on_dismiss=lambda e: print("Dialog dismissed!")
    )

    def handle_menu_item_click(e):
        page.dialog = dlg
        dlg.open = True
        page.update()

    def handle_on_open(e):
        print(f"{e.control.content.value}.on_open")

    def handle_on_close(e):
        print(f"{e.control.content.value}.on_close")

    def handle_on_hover(e):
        print(f"{e.control.content.value}.on_hover")

    def exit_fun(e):
        for filename in temp_filenames:
            if os.path.exists(filename):
                print(f"File {filename} exists.")
                os.remove(filename)
            else:
                print(f"File {filename} does not exist.")
        page.window_destroy()
    
    menubar = ft.MenuBar(
        expand=True,
        style=ft.MenuStyle(
            alignment=ft.alignment.top_left,
            bgcolor= ft.colors.BLUE_400,
            mouse_cursor={ft.MaterialState.HOVERED: ft.MouseCursor.WAIT,
                          ft.MaterialState.DEFAULT: ft.MouseCursor.ZOOM_OUT},
        ),
        controls=[
            ft.SubmenuButton(
                content=ft.Text("File"),
                on_open=handle_on_open,
                on_close=handle_on_close,
                #style=ft.ButtonStyle(bgcolor={ft.MaterialState.HOVERED: ft.colors.GREEN_500}),
                on_hover=handle_on_hover,
                controls=[
                    ft.MenuItemButton(
                        content=ft.Text("About"),
                        leading=ft.Icon(ft.icons.INFO),
                        style=ft.ButtonStyle(bgcolor={ft.MaterialState.HOVERED: ft.colors.GREEN_500}),
                        on_click=handle_menu_item_click
                    ),
                    ft.MenuItemButton(
                        content=ft.Text("Save"),
                        leading=ft.Icon(ft.icons.SAVE),
                        style=ft.ButtonStyle(bgcolor={ft.MaterialState.HOVERED: ft.colors.GREEN_500}),
                        on_click=lambda _: saveme.save_file()
                    ),
                    ft.MenuItemButton(
                        content=ft.Text("Quit"),
                        leading=ft.Icon(ft.icons.CLOSE),
                        style=ft.ButtonStyle(bgcolor={ft.MaterialState.HOVERED: ft.colors.GREEN_500}),
                        on_click=exit_fun
                    )
                ]
            ),
            ft.MenuItemButton(
                        content=ft.Text("Инструкция по работе с программой"),                        
                        #style=ft.ButtonStyle(bgcolor={ft.MaterialState.HOVERED: ft.colors.GREEN_500}),
                        on_click=handle_menu_item_click
                    )
        ]
    )

    def tb3_changes(field, pagee):
        c1.disabled = False
        c2.disabled = False
        c1.update()
        c2.update()
        global last_valid_value
        digits = field.value
        if digits == "":
            last_valid_value = ""
        else:
            try:
                float(digits.replace(",", ""))
                last_valid_value = digits
            except ValueError:
                field.value = last_valid_value
                page.update()

    tb3 = ft.TextField(label="With placeholder", hint_text="Пожалуйста, введите здесь данные",  on_change= lambda e: tb3_changes(tb3, page))
    tb4 = ft.TextField(label="With placeholder", hint_text="Please enter text here",  on_change= lambda e: tb3_changes(tb3, page))

    def c1_click(d):
        print(tb4.value)
        if str(tb3.value) == '' or str(tb4.value) == '':
            print('Пожалуйста, введите')
            return

        np.random.seed(19680801)

        dt = 0.01
        t = np.arange(0, 30, dt)
        nse1 = np.random.randn(len(t))
        nse2 = np.random.randn(len(t))

        s1 = np.sin(float(tb4.value) * np.pi * float(tb3.value) * t) + nse1
        s2 = np.sin(float(tb4.value) * np.pi * float(tb4.value) * t) + nse2

        fig1, ax1 = plt.subplots()  
        ax1.plot(t, s1)
        ax1.set_xlim(0, 2)
        ax1.set_xlabel("time")
        ax1.set_ylabel("s1")
        ax1.grid(True)

        fig2, ax2 = plt.subplots()  
        ax2.plot(t, s2)
        ax2.set_xlim(0, 2)
        ax2.set_xlabel("time")
        ax2.set_ylabel("s2")
        ax2.grid(True)

        fig1.savefig(temp_filenames[0], format="png", bbox_inches="tight", pad_inches=0.1)
        fig2.savefig(temp_filenames[1], format="png", bbox_inches="tight", pad_inches=0.1)
        plt.close(fig1)
        plt.close(fig2)

        page.update()

        chart1.figure = fig1
        chart2.figure = fig2
        
        chart1.update()
        chart2.update()

    c1 = ft.Container(
        content=ft.ElevatedButton("Вычислить"),
        padding=5,
        on_click=c1_click,
        disabled=True       
    )
    c2 = ft.Container(
        content=ft.ElevatedButton("Elevated Button in Container"),
        padding=5,
        on_click=c1_click,
        disabled=True       
    )

    def mysavefile(e:ft.FilePickerResultEvent):
        for temp_filename in temp_filenames:
            if not os.path.exists(temp_filename):
                print(f"Путь к файлу '{temp_filename}' не существует.")
                page.show_snack_bar(ft.SnackBar(content=ft.Text("Ошибка! Графики не были нарисованы.")))
                return
        save_loc = e.path
        if not save_loc:
            print("error save file", e)
            page.show_snack_bar(ft.SnackBar(content=ft.Text("Ошибка! Документ не сохранился.")))
            return
        
        document = Document()
        style = document.styles['Normal']
        style.font.name = 'Times New Roman'
        style.font.size = Pt(14)
        p = document.add_paragraph()
        run = p.add_run("Графики и данные").bold = True

        style.paragraph_format.space_before = Pt(0)
        style.paragraph_format.space_after = Pt(0)
        style.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
        style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        
        document.add_picture(temp_filenames[0])
        style = document.paragraphs[-1]
        style.alignment = 1
        document.add_paragraph("Рисунок 1 – Исходный график с данными")
        style = document.paragraphs[-1]
        style.alignment = 1
        document.add_picture(temp_filenames[1])
        style = document.paragraphs[-1]
        style.alignment = 1

        document.add_paragraph("Рисунок 2 – Исходный график с данными")
        style = document.paragraphs[-1]
        style.alignment = 1
        
        rr = document.add_paragraph(f"Первый набор данных: {float(tb4.value)}")
        rr.paragraph_format.first_line_indent = Cm(1.27)
        
        print(save_loc)
        if '.docx' not in save_loc:
            save_loc = save_loc + '.docx'
        document.save(save_loc)
        
        page.show_snack_bar(ft.SnackBar(content=ft.Text("Графики и данные успешно сохранены в документе Word")))

        page.update()

    saveme = ft.FilePicker(on_result=mysavefile)
    page.overlay.append(saveme)

    fig, ax = plt.subplots()
    chart1 = MatplotlibChart(fig, isolated=True, expand=True)

    chart2 = MatplotlibChart(fig, isolated=True, expand=True)
    plt.close()
    
    page.add(ft.Row([menubar]), tb3, tb4, ft.Row(controls=[c1, c2]), ft.Row([chart1, chart2], width=page.window_width), c)
    
ft.app(target=main)