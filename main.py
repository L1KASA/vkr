from docx import Document
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from flet import Page,TextField
import flet as ft
from flet.matplotlib_chart import MatplotlibChart
import os

matplotlib.use("svg")
temp_filename = "temp_plot.png"
#buf = None

def main(page: ft.Page):
    #buf = None
    def window_event(e):
        print(e.data)
        if e.data == "close":
            print('page closed')
            #buf.close() if buf is not None else page.window_destroy()
            
            if os.path.exists(temp_filename) is True:
                print(12) 
                os.remove(temp_filename) 
            page.window_destroy()
            #page.window_close()

    page.window_prevent_close = True
    page.on_window_event = window_event

    page.theme_mode = ft.ThemeMode.LIGHT
    appbar_text_ref = ft.Ref[ft.Text]()
    
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
        #print(f"{e.control.content.value}.on_click")
        #page.show_snack_bar(ft.SnackBar(content=ft.Text(f"{e.control.content.value} was clicked!")))
        #appbar_text_ref.current.value = e.control.content.value
        #page.update()
        #e.control.page.dialog = dlg
        page.dialog = dlg
        dlg.open = True
        page.update()
        #e.control.page.update()
        

    def handle_on_open(e):
        print(f"{e.control.content.value}.on_open")

    def handle_on_close(e):
        print(f"{e.control.content.value}.on_close")

    def handle_on_hover(e):
        print(f"{e.control.content.value}.on_hover")

    def exit_fun(e):
        if os.path.exists(temp_filename) is True: 
            os.remove(temp_filename)
        page.window_destroy()

    
    menubar = ft.MenuBar(
        expand=True,
        style=ft.MenuStyle(
            alignment=ft.alignment.top_left,
            bgcolor=ft.colors.BLUE_400,
            mouse_cursor={ft.MaterialState.HOVERED: ft.MouseCursor.WAIT,
                          ft.MaterialState.DEFAULT: ft.MouseCursor.ZOOM_OUT},
        ),
        controls=[
            ft.SubmenuButton(
                content=ft.Text("File"),
                on_open=handle_on_open,
                on_close=handle_on_close,
                on_hover=handle_on_hover,
                controls=[
                    ft.MenuItemButton(
                        content=ft.Text("About"),
                        leading=ft.Icon(ft.icons.INFO),
                        style=ft.ButtonStyle(bgcolor={ft.MaterialState.HOVERED: ft.colors.GREEN_500}),
                        on_click=handle_menu_item_click
                        #on_click=dlg
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
                        leading=ft.Icon(ft.icons.INFO),
                        style=ft.ButtonStyle(bgcolor={ft.MaterialState.HOVERED: ft.colors.GREEN_500}),
                        on_click=handle_menu_item_click
                        #on_click=dlg
                    )
        ]
    )

    def button_clicked(e):
        t.value = f"Textboxes values are:  '{tb4.value}'"
        page.update()

    def tb3_changes(field, pagee):
        #print(str(tb4.on_change)[-1])
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
    

    t = ft.Text()
    tb3 = ft.TextField(label="With placeholder", hint_text="Пожалуйста, введите здесь данные",  on_change= lambda e: tb3_changes(tb3, page))
    tb4 = ft.TextField(label="With placeholder", hint_text="Please enter text here",  on_change= lambda e: tb3_changes(tb3, page))
    
    #b = ft.ElevatedButton(text="Submit", on_click=button_clicked)
    #page.add(tb4)

    def c1_click(d):
        print(tb4.value)
        #page.update()
        if str(tb3.value) == '' or str(tb4.value) == '':
            print('Пожалуйста, введите')
            return

        np.random.seed(19680801)

        dt = 0.01
        t = np.arange(0, 30, dt)
        nse1 = np.random.randn(len(t))  # white noise 1
        nse2 = np.random.randn(len(t))  # white noise 2

        # Two signals with a coherent part at 10Hz and a random part
        s1 = np.sin(float(tb4.value) * np.pi * float(tb3.value) * t) + nse1
        s2 = np.sin(float(tb4.value) * np.pi * float(tb4.value) * t) + nse2
       
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(t, s1, t, s2)
        axs[0].set_xlim(0, 2)
        axs[0].set_xlabel("time")
        axs[0].set_ylabel("s1 and s2")
        axs[0].grid(True)

        cxy, f = axs[1].cohere(s1, s2, 256, 1.0 / dt)
        axs[1].set_ylabel("coherence")
        
        fig.tight_layout()

        #plt.close()
        #buf.value = io.BytesIO()
        #fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
        fig.savefig(temp_filename, format="png", bbox_inches="tight", pad_inches=0.1)
        
        #print(buf.value)
        #print(buf)
        #buf.seek(0)
        #print(buf)
        
        plt.close()
        
        #file_picker = ft.FilePicker()
        #page.overlay.append(file_picker)
        #file_picker.save_file()
        #page.update()   
        page.update()
        #ax.legend(title="Fruit color")

    # enable explicit updates
    # and add chart to a page
        #chart1 = MatplotlibChart(fig, isolated=True, expand=True)
        
        #chart1.clean()
        #chart1 = MatplotlibChart(fig, isolated=True, expand=True)
        

        #sleep(5)

        # update chart axis
        #ax.legend(title="Colors")
        chart1.figure = fig
        chart1.update()

       
    c1 = ft.Container(
        content=ft.ElevatedButton("Вычислить"),
        #bgcolor=ft.colors.YELLOW,
        padding=5,
        on_click=c1_click,
        disabled=True       
    )
    c2 = ft.Container(
        content=ft.ElevatedButton("Elevated Button in Container"),
        #bgcolor=ft.colors.YELLOW,
        padding=5,
        on_click=c1_click,
        disabled=True       
    )

    

    def mysavefile(e:ft.FilePickerResultEvent):
        
        print(temp_filename)
        save_loc = e.path
        if not save_loc:
            print("error save file", e)
            return
        
        document = Document()

        # Добавляем заголовок
        document.add_heading("Графики и данные", level=1)
 # Добавляем первый график и надпись
        document.add_heading("График синуса", level=2)
        
        document.add_picture(temp_filename)
        document.add_paragraph(f"Первый набор данных: {float(tb4.value)}")

        
 # Сохраняем документ
        print(save_loc)
        if '.docx' not in save_loc:
            save_loc = save_loc + '.docx'
        document.save(save_loc)
        
        #messagebox.showinfo("Сохранение в docx", "Графики и данные успешно сохранены в документе Word.")
#        
#
        page.update()

    saveme = ft.FilePicker(on_result=mysavefile)
    page.overlay.append(saveme)
           
    # Fixing random state for reproducibility
    #page.add(ft.Row([menubar]), tb3, tb4, ft.Row(controls=[c1, c2]), c)
    #page.add(
    #    ft.Column([
    #        ft.ElevatedButton("Save as file",
    #                       on_click=lambda _: saveme.save_file())
    #    ])
    #)

    fig, ax = plt.subplots(2,1)
    chart1 = MatplotlibChart(fig, isolated=True, expand=True)
    plt.close()
    page.add(ft.Row([menubar]), tb3, tb4, ft.Row(controls=[c1, c2]), chart1, c)
    #page.add(chart1)
    
ft.app(target=main)
#ft.app(target=main, view=ft.AppView.WEB_BROWSER)