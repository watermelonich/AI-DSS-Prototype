import sys
import os
import tkinter as tk
from tkinter import Canvas, Entry, Button, Scrollbar
from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api_wrappers import med, healthsg, fin, newscatcher, supp_chain_opt
from model_wrappers import reg_models_wrapper, clas_models_wrapper


class AIDSSApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI-DSS Sample")
        self.geometry("760x650")
        self.configure(bg="white")

        self.frames = {}
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.create_frames(container)
        self.show_frame("DashboardPage")

    def create_frames(self, container):
        for F in (DashboardPage, FinancePage, HealthPage, MarketingPage, SupplyChainPage, ReportsPage, DataAnalysisPage, ModelSelectionPage, ModelTrainingPage, ModelEvaluationPage):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()

class ScrollableFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="white")
        self.controller = controller

        canvas = tk.Canvas(self, bg="white")
        scrollbar = Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = tk.Frame(canvas, bg="white")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

class SidebarFrame(ScrollableFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.create_sidebar()

    def create_sidebar(self):
        sidebar = tk.Frame(self.scrollable_frame, width=200, bg='#ECF1F6')
        sidebar.pack(expand=False, fill='both', side='left', anchor='nw')

        header_logo_img = tk.PhotoImage(file="../assets/header_logo.png")
        header_logo_img = header_logo_img.subsample(2)
        header_logo_label = tk.Label(sidebar, image=header_logo_img, bg='white')
        header_logo_label.image = header_logo_img
        header_logo_label.pack(side="top", pady=(0, 5))

        label = tk.Label(sidebar, text="Categories", bg='#ECF1F6', font=('Helvetica', 16, 'bold'), fg="#91ABC9")
        label.pack(pady=10)

        buttons = [
            ("Dashboard", lambda: self.controller.show_frame("DashboardPage")),
            ("Reports", lambda: self.controller.show_frame("ReportsPage")),
            ("Data Analysis Tools", lambda: self.controller.show_frame("DataAnalysisPage")),
            ("Model Selection", lambda: self.controller.show_frame("ModelSelectionPage")),
            ("Model Training", lambda: self.controller.show_frame("ModelTrainingPage")),
            ("Model Evaluation", lambda: self.controller.show_frame("ModelEvaluationPage")),
            ("Finance", lambda: self.controller.show_frame("FinancePage")),
            ("Health/Med", lambda: self.controller.show_frame("HealthPage")),
            ("Marketing", lambda: self.controller.show_frame("MarketingPage")),
            ("Supply Chain", lambda: self.controller.show_frame("SupplyChainPage"))
        ]

        for text, command in buttons:
            button = tk.Button(sidebar, text=text, command=command, fg="lightblue", bg='#ECF1F6', bd=0,
                               highlightthickness=0, highlightbackground='#ECF1F6', relief='flat')
            button.pack(fill='x', pady=5)

class DashboardPage(SidebarFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        tk.Label(self.scrollable_frame, text="Dashboard", font=('Helvetica', 18), bg="white", fg="#4974a5").pack(pady=20)

        profile_menu_frame = tk.Frame(self.scrollable_frame, bg="white")
        profile_menu_frame.place(relx=1.0, rely=0, anchor='ne', x=-20, y=20)

        profile_icon_img = tk.PhotoImage(file="../assets/profile_icon.png")
        profile_icon_label = tk.Label(profile_menu_frame, image=profile_icon_img, bg="white")
        profile_icon_label.image = profile_icon_img
        profile_icon_label.pack(side="left")

        menu_bar_img = tk.PhotoImage(file="../assets/menu.png")
        menu_bar_label = tk.Label(profile_menu_frame, image=menu_bar_img, bg="white")
        menu_bar_label.image = menu_bar_img
        menu_bar_label.pack(side="left", padx=10)

        swit_logo_img = tk.PhotoImage(file="../assets/swit_logo.png")
        swit_logo_label = tk.Label(profile_menu_frame, image=swit_logo_img, bg="white", cursor="hand2")
        swit_logo_label.image = swit_logo_img
        swit_logo_label.pack(side="right")
        swit_logo_label.bind("<Button-1>", lambda event: controller.show_frame("ChatPage"))

        self.create_dashboard_plots()

    def create_dashboard_plots(self):
        plots = [
            ("Dashboard Plot 1", self.create_line_plot),
            ("Dashboard Plot 2", self.create_bar_plot),
            ("Dashboard Plot 3", self.create_pie_chart),
            ("Dashboard Plot 4", self.create_scatter_plot)
        ]
        for title, plot_func in plots:
            self.create_plot(title, plot_func)

    def create_plot(self, title, plot_func):
        frame = tk.Frame(self.scrollable_frame, bg="white")
        frame.pack(fill='both', expand=True, pady=10)
        label = tk.Label(frame, text=title, font=('Helvetica', 12), bg="white")
        label.pack()
        figure = plot_func()
        canvas = FigureCanvasTkAgg(figure, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def create_line_plot(self):
        figure = Figure(figsize=(5, 4), dpi=100)
        ax = figure.add_subplot(111)
        ax.plot([1, 2, 3, 4, 5], [1, 4, 2, 5, 3])
        return figure

    def create_bar_plot(self):
        figure = Figure(figsize=(5, 4), dpi=100)
        ax = figure.add_subplot(111)
        ax.bar(['A', 'B', 'C', 'D'], [5, 3, 7, 2])
        return figure

    def create_pie_chart(self):
        figure = Figure(figsize=(5, 4), dpi=100)
        ax = figure.add_subplot(111)
        ax.pie([10, 20, 30, 40], labels=['Q1', 'Q2', 'Q3', 'Q4'], autopct='%1.1f%%')
        return figure

    def create_scatter_plot(self):
        figure = Figure(figsize=(5, 4), dpi=100)
        ax = figure.add_subplot(111)
        ax.scatter([1, 2, 3, 4, 5], [5, 2, 4, 7, 1])
        return figure

class FinancePage(SidebarFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.create_finance_panel()

    def create_finance_panel(self):
        frame = tk.Frame(self.scrollable_frame, pady=0)
        frame.pack(fill='x')
        label = tk.Label(frame, text="Finchat Data", font=('Helvetica', 14))
        label.pack(side='left')
        input_entry = tk.Entry(frame)
        input_entry.pack(side='left', padx=5)
        button = tk.Button(frame, text="Get Finance Data", command=lambda: self.fetch_finchat_data(input_entry))
        button.pack(side='left', padx=5)
        self.finchat_result = tk.Text(self.scrollable_frame, height=5, wrap='word')
        self.finchat_result.pack(fill='x', padx=5, pady=5)

        self.create_finance_plots()

    def fetch_finchat_data(self, entry):
        params = {"param": entry.get()}
        data = fin.get_financial_data(params)
        self.finchat_result.delete(1.0, tk.END)
        self.finchat_result.insert(tk.END, str(data))

    def create_finance_plots(self):
        plots = [
            ("Finance Plot 1", self.create_line_plot),
            ("Finance Plot 2", self.create_bar_plot),
            ("Finance Plot 3", self.create_pie_chart),
            ("Finance Plot 4", self.create_scatter_plot)
        ]
        for title, plot_func in plots:
            self.create_plot(title, plot_func)

    def create_plot(self, title, plot_func):
        frame = tk.Frame(self.scrollable_frame, bg="white")
        frame.pack(fill='both', expand=True, pady=10)
        label = tk.Label(frame, text=title, font=('Helvetica', 12), bg="white")
        label.pack()
        figure = plot_func()
        canvas = FigureCanvasTkAgg(figure, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def create_line_plot(self):
        figure = Figure(figsize=(5, 4), dpi=100)
        ax = figure.add_subplot(111)
        ax.plot([1, 2, 3, 4, 5], [1, 4, 2, 5, 3])
        return figure

    def create_bar_plot(self):
        figure = Figure(figsize=(5, 4), dpi=100)
        ax = figure.add_subplot(111)
        ax.bar(['A', 'B', 'C', 'D'], [5, 3, 7, 2])
        return figure

    def create_pie_chart(self):
        figure = Figure(figsize=(5, 4), dpi=100)
        ax = figure.add_subplot(111)
        ax.pie([10, 20, 30, 40], labels=['Q1', 'Q2', 'Q3', 'Q4'], autopct='%1.1f%%')
        return figure

    def create_scatter_plot(self):
        figure = Figure(figsize=(5, 4), dpi=100)
        ax = figure.add_subplot(111)
        ax.scatter([1, 2, 3, 4, 5], [5, 2, 4, 7, 1])
        return figure

class HealthPage(SidebarFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.create_health_panel()

    def create_health_panel(self):
        frame1 = tk.Frame(self.scrollable_frame, pady=10)
        frame1.pack(fill='x')
        label1 = tk.Label(frame1, text="Endless Medical Data", font=('Helvetica', 14))
        label1.pack(side='left')
        input_entry1 = tk.Entry(frame1)
        input_entry1.pack(side='left', padx=5)
        button1 = tk.Button(frame1, text="Get Medical Data", command=lambda: self.fetch_medical_data(input_entry1))
        button1.pack(side='left', padx=5)
        self.medical_result = tk.Text(self.scrollable_frame, height=5, wrap='word')
        self.medical_result.pack(fill='x', padx=5, pady=5)

        frame2 = tk.Frame(self.scrollable_frame, pady=10)
        frame2.pack(fill='x')
        label2 = tk.Label(frame2, text="Healthsage Data", font=('Helvetica', 14))
        label2.pack(side='left')
        input_entry2 = tk.Entry(frame2)
        input_entry2.pack(side='left', padx=5)
        button2 = tk.Button(frame2, text="Get Healthsage Data", command=lambda: self.fetch_healthsage_data(input_entry2))
        button2.pack(side='left', padx=5)
        self.healthsage_result = tk.Text(self.scrollable_frame, height=5, wrap='word')
        self.healthsage_result.pack(fill='x', padx=5, pady=5)

        self.create_health_plots()

    def fetch_medical_data(self, entry):
        params = {"param": entry.get()}
        data = med.get_medical_data(params)
        self.medical_result.delete(1.0, tk.END)
        self.medical_result.insert(tk.END, str(data))

    def fetch_healthsage_data(self, entry):
        params = {"param": entry.get()}
        data = healthsg.get_healthsage_data(params)
        self.healthsage_result.delete(1.0, tk.END)
        self.healthsage_result.insert(tk.END, str(data))

    def create_health_plots(self):
        plots = [
            ("Health Plot 1", self.create_line_plot),
            ("Health Plot 2", self.create_bar_plot),
            ("Health Plot 3", self.create_pie_chart),
            ("Health Plot 4", self.create_scatter_plot)
        ]
        for title, plot_func in plots:
            self.create_plot(title, plot_func)

    def create_plot(self, title, plot_func):
        frame = tk.Frame(self.scrollable_frame, bg="white")
        frame.pack(fill='both', expand=True, pady=10)
        label = tk.Label(frame, text=title, font=('Helvetica', 12), bg="white")
        label.pack()
        figure = plot_func()
        canvas = FigureCanvasTkAgg(figure, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def create_line_plot(self):
        figure = Figure(figsize=(5, 4), dpi=100)
        ax = figure.add_subplot(111)
        ax.plot([1, 2, 3, 4, 5], [1, 4, 2, 5, 3])
        return figure

    def create_bar_plot(self):
        figure = Figure(figsize=(5, 4), dpi=100)
        ax = figure.add_subplot(111)
        ax.bar(['A', 'B', 'C', 'D'], [5, 3, 7, 2])
        return figure

    def create_pie_chart(self):
        figure = Figure(figsize=(5, 4), dpi=100)
        ax = figure.add_subplot(111)
        ax.pie([10, 20, 30, 40], labels=['Q1', 'Q2', 'Q3', 'Q4'], autopct='%1.1f%%')
        return figure

    def create_scatter_plot(self):
        figure = Figure(figsize=(5, 4), dpi=100)
        ax = figure.add_subplot(111)
        ax.scatter([1, 2, 3, 4, 5], [5, 2, 4, 7, 1])
        return figure

class MarketingPage(SidebarFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.create_marketing_panel()

    def create_marketing_panel(self):
        frame = tk.Frame(self.scrollable_frame, pady=10)
        frame.pack(fill='x')
        label = tk.Label(frame, text="Newscatcher Data", font=('Helvetica', 14))
        label.pack(side='left')
        input_entry = tk.Entry(frame)
        input_entry.pack(side='left', padx=5)
        button = tk.Button(frame, text="Get Marketing Data", command=lambda: self.fetch_newscatcher_data(input_entry))
        button.pack(side='left', padx=5)
        self.newscatcher_result = tk.Text(self.scrollable_frame, height=10, wrap='word')
        self.newscatcher_result.pack(fill='x', padx=5, pady=5)

        self.create_marketing_plots()

    def fetch_newscatcher_data(self, entry):
        params = {"param": entry.get()}
        data = newscatcher.get_news_data(params)
        self.newscatcher_result.delete(1.0, tk.END)
        self.newscatcher_result.insert(tk.END, str(data))

    def create_marketing_plots(self):
        plots = [
            ("Marketing Plot 1", self.create_line_plot),
            ("Marketing Plot 2", self.create_bar_plot),
            ("Marketing Plot 3", self.create_pie_chart),
            ("Marketing Plot 4", self.create_scatter_plot)
        ]
        for title, plot_func in plots:
            self.create_plot(title, plot_func)

    def create_plot(self, title, plot_func):
        frame = tk.Frame(self.scrollable_frame, bg="white")
        frame.pack(fill='both', expand=True, pady=10)
        label = tk.Label(frame, text=title, font=('Helvetica', 12), bg="white")
        label.pack()
        figure = plot_func()
        canvas = FigureCanvasTkAgg(figure, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def create_line_plot(self):
        figure = Figure(figsize=(5, 4), dpi=100)
        ax = figure.add_subplot(111)
        ax.plot([1, 2, 3, 4, 5], [1, 4, 2, 5, 3])
        return figure

    def create_bar_plot(self):
        figure = Figure(figsize=(5, 4), dpi=100)
        ax = figure.add_subplot(111)
        ax.bar(['A', 'B', 'C', 'D'], [5, 3, 7, 2])
        return figure

    def create_pie_chart(self):
        figure = Figure(figsize=(5, 4), dpi=100)
        ax = figure.add_subplot(111)
        ax.pie([10, 20, 30, 40], labels=['Q1', 'Q2', 'Q3', 'Q4'], autopct='%1.1f%%')
        return figure

    def create_scatter_plot(self):
        figure = Figure(figsize=(5, 4), dpi=100)
        ax = figure.add_subplot(111)
        ax.scatter([1, 2, 3, 4, 5], [5, 2, 4, 7, 1])
        return figure

class SupplyChainPage(SidebarFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.create_supply_chain_panel()

    def create_supply_chain_panel(self):
        frame = tk.Frame(self.scrollable_frame, pady=10)
        frame.pack(fill='x')
        label = tk.Label(frame, text="Supply Chain Optimizer Data", font=('Helvetica', 14))
        label.pack(side='left')
        input_entry = tk.Entry(frame)
        input_entry.pack(side='left', padx=5)
        button = tk.Button(frame, text="Get Supply Chain Data", command=lambda: self.fetch_supply_chain_data(input_entry))
        button.pack(side='left', padx=5)
        self.supply_chain_result = tk.Text(self.scrollable_frame, height=10, wrap='word')
        self.supply_chain_result.pack(fill='x', padx=5, pady=5)

        self.create_supply_chain_plots()

    def fetch_supply_chain_data(self, entry):
        params = {"param": entry.get()}
        data = supp_chain_opt.get_supply_chain_data(params)
        self.supply_chain_result.delete(1.0, tk.END)
        self.supply_chain_result.insert(tk.END, str(data))

    def create_supply_chain_plots(self):
        plots = [
            ("Supply Chain Plot 1", self.create_line_plot),
            ("Supply Chain Plot 2", self.create_bar_plot),
            ("Supply Chain Plot 3", self.create_pie_chart),
            ("Supply Chain Plot 4", self.create_scatter_plot)
        ]
        for title, plot_func in plots:
            self.create_plot(title, plot_func)

    def create_plot(self, title, plot_func):
        frame = tk.Frame(self.scrollable_frame, bg="white")
        frame.pack(fill='both', expand=True, pady=10)
        label = tk.Label(frame, text=title, font=('Helvetica', 12), bg="white")
        label.pack()
        figure = plot_func()
        canvas = FigureCanvasTkAgg(figure, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def create_line_plot(self):
        figure = Figure(figsize=(5, 4), dpi=100)
        ax = figure.add_subplot(111)
        ax.plot([1, 2, 3, 4, 5], [1, 4, 2, 5, 3])
        return figure

    def create_bar_plot(self):
        figure = Figure(figsize=(5, 4), dpi=100)
        ax = figure.add_subplot(111)
        ax.bar(['A', 'B', 'C', 'D'], [5, 3, 7, 2])
        return figure

    def create_pie_chart(self):
        figure = Figure(figsize=(5, 4), dpi=100)
        ax = figure.add_subplot(111)
        ax.pie([10, 20, 30, 40], labels=['Q1', 'Q2', 'Q3', 'Q4'], autopct='%1.1f%%')
        return figure

    def create_scatter_plot(self):
        figure = Figure(figsize=(5, 4), dpi=100)
        ax = figure.add_subplot(111)
        ax.scatter([1, 2, 3, 4, 5], [5, 2, 4, 7, 1])
        return figure

class ReportsPage(SidebarFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        tk.Label(self.scrollable_frame, text="Reports", font=('Helvetica', 18), bg="white", fg="#4974a5").pack(pady=20)

class DataAnalysisPage(SidebarFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        tk.Label(self.scrollable_frame, text="Data Analysis Tools", font=('Helvetica', 18), bg="white", fg="#4974a5").pack(pady=20)

class ModelSelectionPage(SidebarFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        tk.Label(self.scrollable_frame, text="Model Selection", font=('Helvetica', 18), bg="white", fg="#4974a5").pack(pady=20)

class ModelTrainingPage(SidebarFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        tk.Label(self.scrollable_frame, text="Model Training", font=('Helvetica', 18), bg="white", fg="#4974a5").pack(pady=20)

class ModelEvaluationPage(SidebarFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        tk.Label(self.scrollable_frame, text="Model Evaluation", font=('Helvetica', 18), bg="white", fg="#4974a5").pack(pady=20)

class ChatPage(SidebarFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        tk.Label(self.scrollable_frame, text="Chat", font=('Helvetica', 18), bg="white", fg="#4974a5").pack(pady=20)

        back_button = tk.Button(self.scrollable_frame, text="Back", command=lambda: controller.show_frame("DashboardPage"))
        back_button.pack()

if __name__ == "__main__":
    app = AIDSSApp()
    app.mainloop()
