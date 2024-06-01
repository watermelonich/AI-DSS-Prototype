# import sys
# import os

# sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# import tkinter as tk
# from tkinter import Canvas
# from PIL import Image, ImageTk
# from api_wrappers import med, healthsg, fin, newscatcher, supp_chain_opt
# from model_wrappers import reg_models_wrapper, clas_models_wrapper, earth_mapplot_wrapper


# class AIDSSApp(tk.Tk):
#     def __init__(self):
#         super().__init__()
#         self.title("AI-DSS Sample")
#         self.geometry("880x610")
#         self.configure(bg="white")  # Set background color to white

#         self.frames = {}
#         container = tk.Frame(self)
#         container.pack(side="top", fill="both", expand=True)
#         container.grid_rowconfigure(0, weight=1)
#         container.grid_columnconfigure(0, weight=1)

#         self.create_frames(container)
#         self.show_frame("HomePage")

#     def create_frames(self, container):
#         self.frames["HomePage"] = HomePage(parent=container, controller=self)
#         self.frames["FinancePage"] = FinancePage(parent=container, controller=self)
#         self.frames["HealthPage"] = HealthPage(parent=container, controller=self)
#         self.frames["MarketingPage"] = MarketingPage(parent=container, controller=self)
#         self.frames["SupplyChainPage"] = SupplyChainPage(parent=container, controller=self)

#         for frame in self.frames.values():
#             frame.grid(row=0, column=0, sticky="nsew")

#     def show_frame(self, page_name):
#         frame = self.frames[page_name]
#         frame.tkraise()

# class HomePage(tk.Frame):
#     def __init__(self, parent, controller):
#         super().__init__(parent, bg="white")
#         self.controller = controller
#         tk.Label(self, text="Welcome back, [User]!", font=('Helvetica', 18), bg="white", fg="#4974a5").pack(pady=20)
#         self.create_sidebar()

#     def create_sidebar(self):
#         sidebar = tk.Frame(self, width=200, bg='#ECF1F6')
#         sidebar.pack(expand=False, fill='both', side='left', anchor='nw')

#         canvas = Canvas(sidebar, width=200, height=200, bg='#ECF1F6', highlightthickness=0)
#         canvas.pack(fill='both', expand=True)

#         r = 80
#         canvas.create_arc((0, 0, r*2, r*2), start=90, extent=90, fill='#ECF1F6', outline='#ECF1F6')
#         canvas.create_arc((200-r*2, 0, 200, r*2), start=0, extent=90, fill='#ECF1F6', outline='#ECF1F6')
#         canvas.create_rectangle((r, 0, 200-r, r), fill='#ECF1F6', outline='#ECF1F6')
#         canvas.create_rectangle((0, r, 200, 200), fill='#ECF1F6', outline='#ECF1F6')

#         label = tk.Label(canvas, text="Categories", bg='#ECF1F6', font=('Helvetica', 16, 'bold'), fg="#91ABC9")
#         canvas.create_window(100, 50, window=label)

#         buttons = [
#             ("Dashboard", lambda: self.controller.show_frame("HomePage")),
#             ("Finance", lambda: self.controller.show_frame("FinancePage")),
#             ("Health/Med", lambda: self.controller.show_frame("HealthPage")),
#             ("Marketing", lambda: self.controller.show_frame("MarketingPage")),
#             ("Supply Chain", lambda: self.controller.show_frame("SupplyChainPage"))
#         ]
        
#         for i, (text, command) in enumerate(buttons):
#             button = tk.Button(canvas, text=text, command=command, fg="lightblue", bg='#ECF1F6', bd=0, highlightthickness=0, highlightbackground='#ECF1F6', relief='flat')
#             canvas.create_window(100, 100 + i * 40, window=button, anchor='n', width=180)


# class FinancePage(tk.Frame):
#     def __init__(self, parent, controller):
#         super().__init__(parent, bg="white")
#         self.controller = controller
#         self.create_sidebar()
#         self.create_finance_panel()

#     def create_sidebar(self):
#         sidebar = tk.Frame(self, width=200, bg='#ECF1F6')
#         sidebar.pack(expand=False, fill='both', side='left', anchor='nw')
#         tk.Label(sidebar, text="Categories", bg='#ECF1F6', font=('Helvetica', 16, 'bold')).pack(pady=10)

#         tk.Button(sidebar, text="Home", command=lambda: self.controller.show_frame("HomePage"), bd=0, highlightthickness=0, highlightbackground='#ECF1F6', relief='flat').pack(fill='x')
#         tk.Button(sidebar, text="Finance", command=lambda: self.controller.show_frame("FinancePage"), bd=0, highlightthickness=0, highlightbackground='#ECF1F6', relief='flat').pack(fill='x')
#         tk.Button(sidebar, text="Health/Med", command=lambda: self.controller.show_frame("HealthPage"), bd=0, highlightthickness=0, highlightbackground='#ECF1F6', relief='flat').pack(fill='x')
#         tk.Button(sidebar, text="Marketing", command=lambda: self.controller.show_frame("MarketingPage"), bd=0, highlightthickness=0, highlightbackground='#ECF1F6', relief='flat').pack(fill='x')
#         tk.Button(sidebar, text="Supply Chain", command=lambda: self.controller.show_frame("SupplyChainPage"), bd=0, highlightthickness=0, highlightbackground='#ECF1F6', relief='flat').pack(fill='x')

#     def create_finance_panel(self):
#         frame = tk.Frame(self, pady=10)
#         frame.pack(fill='x')
#         label = tk.Label(frame, text="Finchat Data", font=('Helvetica', 14))
#         label.pack(side='left')
#         input_entry = tk.Entry(frame)
#         input_entry.pack(side='left', padx=5)
#         button = tk.Button(frame, text="Get Finance Data", command=lambda: self.fetch_finchat_data(input_entry))
#         button.pack(side='left', padx=5)
#         self.finchat_result = tk.Text(self, height=5, wrap='word')
#         self.finchat_result.pack(fill='x', padx=5, pady=5)

#     def fetch_finchat_data(self, entry):
#         params = {"param": entry.get()}
#         data = fin.get_financial_data(params)
#         self.finchat_result.delete(1.0, tk.END)
#         self.finchat_result.insert(tk.END, str(data))

# class HealthPage(tk.Frame):
#     def __init__(self, parent, controller):
#         super().__init__(parent, bg="white")
#         self.controller = controller
#         self.create_sidebar()
#         self.create_health_panel()

#     def create_sidebar(self):
#         sidebar = tk.Frame(self, width=200, bg='#ECF1F6')
#         sidebar.pack(expand=False, fill='both', side='left', anchor='nw')
#         tk.Label(sidebar, text="Categories", bg='#ECF1F6', font=('Helvetica', 16, 'bold')).pack(pady=10)

#         tk.Button(sidebar, text="Home", command=lambda: self.controller.show_frame("HomePage"), bd=0, highlightthickness=0, highlightbackground='#ECF1F6', relief='flat').pack(fill='x')
#         tk.Button(sidebar, text="Finance", command=lambda: self.controller.show_frame("FinancePage"), bd=0, highlightthickness=0, highlightbackground='#ECF1F6', relief='flat').pack(fill='x')
#         tk.Button(sidebar, text="Health/Med", command=lambda: self.controller.show_frame("HealthPage"), bd=0, highlightthickness=0, highlightbackground='#ECF1F6', relief='flat').pack(fill='x')
#         tk.Button(sidebar, text="Marketing", command=lambda: self.controller.show_frame("MarketingPage"), bd=0, highlightthickness=0, highlightbackground='#ECF1F6', relief='flat').pack(fill='x')
#         tk.Button(sidebar, text="Supply Chain", command=lambda: self.controller.show_frame("SupplyChainPage"), bd=0, highlightthickness=0, highlightbackground='#ECF1F6', relief='flat').pack(fill='x')

#     def create_health_panel(self):
#         frame1 = tk.Frame(self, pady=10)
#         frame1.pack(fill='x')
#         label1 = tk.Label(frame1, text="Endless Medical Data", font=('Helvetica', 14))
#         label1.pack(side='left')
#         input_entry1 = tk.Entry(frame1)
#         input_entry1.pack(side='left', padx=5)
#         button1 = tk.Button(frame1, text="Get Medical Data", command=lambda: self.fetch_medical_data(input_entry1))
#         button1.pack(side='left', padx=5)
#         self.medical_result = tk.Text(self, height=5, wrap='word')
#         self.medical_result.pack(fill='x', padx=5, pady=5)

#         frame2 = tk.Frame(self, pady=10)
#         frame2.pack(fill='x')
#         label2 = tk.Label(frame2, text="Healthsage Data", font=('Helvetica', 14))
#         label2.pack(side='left')
#         input_entry2 = tk.Entry(frame2)
#         input_entry2.pack(side='left', padx=5)
#         button2 = tk.Button(frame2, text="Get Healthsage Data", command=lambda: self.fetch_healthsage_data(input_entry2))
#         button2.pack(side='left', padx=5)
#         self.healthsage_result = tk.Text(self, height=5, wrap='word')
#         self.healthsage_result.pack(fill='x', padx=5, pady=5)

#     def fetch_medical_data(self, entry):
#         params = {"param": entry.get()}
#         data = med.get_medical_data(params)
#         self.medical_result.delete(1.0, tk.END)
#         self.medical_result.insert(tk.END, str(data))

#     def fetch_healthsage_data(self, entry):
#         params = {"param": entry.get()}
#         data = healthsg.get_healthsage_data(params)
#         self.healthsage_result.delete(1.0, tk.END)
#         self.healthsage_result.insert(tk.END, str(data))

# class MarketingPage(tk.Frame):
#     def __init__(self, parent, controller):
#         super().__init__(parent, bg="white")
#         self.controller = controller
#         self.create_sidebar()
#         self.create_marketing_panel()

#     def create_sidebar(self):
#         sidebar = tk.Frame(self, width=200, bg='#ECF1F6')
#         sidebar.pack(expand=False, fill='both', side='left', anchor='nw')
#         tk.Label(sidebar, text="Categories", bg='#ECF1F6', font=('Helvetica', 16, 'bold')).pack(pady=10)

#         tk.Button(sidebar, text="Home", command=lambda: self.controller.show_frame("HomePage"), bd=0, highlightthickness=0, highlightbackground='#ECF1F6', relief='flat').pack(fill='x')
#         tk.Button(sidebar, text="Finance", command=lambda: self.controller.show_frame("FinancePage"), bd=0, highlightthickness=0, highlightbackground='#ECF1F6', relief='flat').pack(fill='x')
#         tk.Button(sidebar, text="Health/Med", command=lambda: self.controller.show_frame("HealthPage"), bd=0, highlightthickness=0, highlightbackground='#ECF1F6', relief='flat').pack(fill='x')
#         tk.Button(sidebar, text="Marketing", command=lambda: self.controller.show_frame("MarketingPage"), bd=0, highlightthickness=0, highlightbackground='#ECF1F6', relief='flat').pack(fill='x')
#         tk.Button(sidebar, text="Supply Chain", command=lambda: self.controller.show_frame("SupplyChainPage"), bd=0, highlightthickness=0, highlightbackground='#ECF1F6', relief='flat').pack(fill='x')

#     def create_marketing_panel(self):
#         frame = tk.Frame(self, pady=10)
#         frame.pack(fill='x')
#         frame = tk.Frame(self, pady=10)
#         frame.pack(fill='x')
#         label = tk.Label(frame, text="Newscatcher Data", font=('Helvetica', 14))
#         label.pack(side='left')
#         input_entry = tk.Entry(frame)
#         input_entry.pack(side='left', padx=5)
#         button = tk.Button(frame, text="Get Marketing Data", command=lambda: self.fetch_newscatcher_data(input_entry))
#         button.pack(side='left', padx=5)
#         self.newscatcher_result = tk.Text(self, height=10, wrap='word')
#         self.newscatcher_result.pack(fill='x', padx=5, pady=5)

#     def fetch_newscatcher_data(self, entry):
#         params = {"param": entry.get()}
#         data = newscatcher.get_news_data(params)
#         self.newscatcher_result.delete(1.0, tk.END)
#         self.newscatcher_result.insert(tk.END, str(data))

# class SupplyChainPage(tk.Frame):
#     def __init__(self, parent, controller):
#         super().__init__(parent, bg="white")
#         self.controller = controller
#         self.create_sidebar()
#         self.create_supply_chain_panel()

#     def create_sidebar(self):
#         sidebar = tk.Frame(self, width=200, bg='#ECF1F6')
#         sidebar.pack(expand=False, fill='both', side='left', anchor='nw')
#         tk.Label(sidebar, text="Categories", bg='#ECF1F6', font=('Helvetica', 16, 'bold')).pack(pady=10)

#         tk.Button(sidebar, text="Home", command=lambda: self.controller.show_frame("HomePage"), bd=0, highlightthickness=0, highlightbackground='#ECF1F6', relief='flat').pack(fill='x')
#         tk.Button(sidebar, text="Finance", command=lambda: self.controller.show_frame("FinancePage"), bd=0, highlightthickness=0, highlightbackground='#ECF1F6', relief='flat').pack(fill='x')
#         tk.Button(sidebar, text="Health/Med", command=lambda: self.controller.show_frame("HealthPage"), bd=0, highlightthickness=0, highlightbackground='#ECF1F6', relief='flat').pack(fill='x')
#         tk.Button(sidebar, text="Marketing", command=lambda: self.controller.show_frame("MarketingPage"), bd=0, highlightthickness=0, highlightbackground='#ECF1F6', relief='flat').pack(fill='x')
#         tk.Button(sidebar, text="Supply Chain", command=lambda: self.controller.show_frame("SupplyChainPage"), bd=0, highlightthickness=0, highlightbackground='#ECF1F6', relief='flat').pack(fill='x')

#     def create_supply_chain_panel(self):
#         frame = tk.Frame(self, pady=10)
#         frame.pack(fill='x')
#         label = tk.Label(frame, text="Supply Chain Optimizer Data", font=('Helvetica', 14))
#         label.pack(side='left')
#         input_entry = tk.Entry(frame)
#         input_entry.pack(side='left', padx=5)
#         button = tk.Button(frame, text="Get Supply Chain Data", command=lambda: self.fetch_supply_chain_data(input_entry))
#         button.pack(side='left', padx=5)
#         self.supply_chain_result = tk.Text(self, height=10, wrap='word')
#         self.supply_chain_result.pack(fill='x', padx=5, pady=5)

#     def fetch_supply_chain_data(self, entry):
#         params = {"param": entry.get()}
#         data = supp_chain_opt.get_supply_chain_data(params)
#         self.supply_chain_result.delete(1.0, tk.END)
#         self.supply_chain_result.insert(tk.END, str(data))

# if __name__ == "__main__":
#     app = AIDSSApp()
#     app.mainloop()

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import tkinter as tk
from tkinter import Canvas
from PIL import Image, ImageTk
from api_wrappers import med, healthsg, fin, newscatcher, supp_chain_opt
from model_wrappers import reg_models_wrapper, clas_models_wrapper


class AIDSSApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI-DSS Sample")
        self.geometry("880x610")
        self.configure(bg="white")  # Set background color to white

        self.frames = {}
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.create_frames(container)
        self.show_frame("HomePage")

    def create_frames(self, container):
        for F in (HomePage, FinancePage, HealthPage, MarketingPage, SupplyChainPage, DashboardPage, ReportsPage, DataAnalysisPage, ModelSelectionPage, ModelTrainingPage, ModelEvaluationPage):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()


class SidebarFrame(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg="white")
        self.controller = controller
        self.create_sidebar()

    def create_sidebar(self):
        sidebar = tk.Frame(self, width=200, bg='#ECF1F6')
        sidebar.pack(expand=False, fill='both', side='left', anchor='nw')

        canvas = Canvas(sidebar, width=200, height=200, bg='#ECF1F6', highlightthickness=0)
        canvas.pack(fill='both', expand=True)

        r = 80
        canvas.create_arc((0, 0, r * 2, r * 2), start=90, extent=90, fill='#ECF1F6', outline='#ECF1F6')
        canvas.create_arc((200 - r * 2, 0, 200, r * 2), start=0, extent=90, fill='#ECF1F6', outline='#ECF1F6')
        canvas.create_rectangle((r, 0, 200 - r, r), fill='#ECF1F6', outline='#ECF1F6')
        canvas.create_rectangle((0, r, 200, 200), fill='#ECF1F6', outline='#ECF1F6')

        label = tk.Label(canvas, text="Categories", bg='#ECF1F6', font=('Helvetica', 16, 'bold'), fg="#91ABC9")
        canvas.create_window(100, 50, window=label)

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

        for i, (text, command) in enumerate(buttons):
            button = tk.Button(canvas, text=text, command=command, fg="lightblue", bg='#ECF1F6', bd=0,
                               highlightthickness=0, highlightbackground='#ECF1F6', relief='flat')
            canvas.create_window(100, 100 + i * 40, window=button, anchor='n', width=180)


class HomePage(SidebarFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        tk.Label(self, text="Welcome back, [User]!", font=('Helvetica', 18), bg="white", fg="#4974a5").pack(pady=20)


class FinancePage(SidebarFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.create_finance_panel()

    def create_finance_panel(self):
        frame = tk.Frame(self, pady=10)
        frame.pack(fill='x')
        label = tk.Label(frame, text="Finchat Data", font=('Helvetica', 14))
        label.pack(side='left')
        input_entry = tk.Entry(frame)
        input_entry.pack(side='left', padx=5)
        button = tk.Button(frame, text="Get Finance Data", command=lambda: self.fetch_finchat_data(input_entry))
        button.pack(side='left', padx=5)
        self.finchat_result = tk.Text(self, height=5, wrap='word')
        self.finchat_result.pack(fill='x', padx=5, pady=5)

    def fetch_finchat_data(self, entry):
        params = {"param": entry.get()}
        data = fin.get_financial_data(params)
        self.finchat_result.delete(1.0, tk.END)
        self.finchat_result.insert(tk.END, str(data))


class HealthPage(SidebarFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.create_health_panel()

    def create_health_panel(self):
        frame1 = tk.Frame(self, pady=10)
        frame1.pack(fill='x')
        label1 = tk.Label(frame1, text="Endless Medical Data", font=('Helvetica', 14))
        label1.pack(side='left')
        input_entry1 = tk.Entry(frame1)
        input_entry1.pack(side='left', padx=5)
        button1 = tk.Button(frame1, text="Get Medical Data", command=lambda: self.fetch_medical_data(input_entry1))
        button1.pack(side='left', padx=5)
        self.medical_result = tk.Text(self, height=5, wrap='word')
        self.medical_result.pack(fill='x', padx=5, pady=5)

        frame2 = tk.Frame(self, pady=10)
        frame2.pack(fill='x')
        label2 = tk.Label(frame2, text="Healthsage Data", font=('Helvetica', 14))
        label2.pack(side='left')
        input_entry2 = tk.Entry(frame2)
        input_entry2.pack(side='left', padx=5)
        button2 = tk.Button(frame2, text="Get Healthsage Data", command=lambda: self.fetch_healthsage_data(input_entry2))
        button2.pack(side='left', padx=5)
        self.healthsage_result = tk.Text(self, height=5, wrap='word')
        self.healthsage_result.pack(fill='x', padx=5, pady=5)

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


class MarketingPage(SidebarFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.create_marketing_panel()

    def create_marketing_panel(self):
        frame = tk.Frame(self, pady=10)
        frame.pack(fill='x')
        label = tk.Label(frame, text="Newscatcher Data", font=('Helvetica', 14))
        label.pack(side='left')
        input_entry = tk.Entry(frame)
        input_entry.pack(side='left', padx=5)
        button = tk.Button(frame, text="Get Marketing Data", command=lambda: self.fetch_newscatcher_data(input_entry))
        button.pack(side='left', padx=5)
        self.newscatcher_result = tk.Text(self, height=10, wrap='word')
        self.newscatcher_result.pack(fill='x', padx=5, pady=5)

    def fetch_newscatcher_data(self, entry):
        params = {"param": entry.get()}
        data = newscatcher.get_news_data(params)
        self.newscatcher_result.delete(1.0, tk.END)
        self.newscatcher_result.insert(tk.END, str(data))


class SupplyChainPage(SidebarFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.create_supply_chain_panel()

    def create_supply_chain_panel(self):
        frame = tk.Frame(self, pady=10)
        frame.pack(fill='x')
        label = tk.Label(frame, text="Supply Chain Optimizer Data", font=('Helvetica', 14))
        label.pack(side='left')
        input_entry = tk.Entry(frame)
        input_entry.pack(side='left', padx=5)
        button = tk.Button(frame, text="Get Supply Chain Data", command=lambda: self.fetch_supply_chain_data(input_entry))
        button.pack(side='left', padx=5)
        self.supply_chain_result = tk.Text(self, height=10, wrap='word')
        self.supply_chain_result.pack(fill='x', padx=5, pady=5)

    def fetch_supply_chain_data(self, entry):
        params = {"param": entry.get()}
        data = supp_chain_opt.get_supply_chain_data(params)
        self.supply_chain_result.delete(1.0, tk.END)
        self.supply_chain_result.insert(tk.END, str(data))


class DashboardPage(SidebarFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        tk.Label(self, text="Dashboard", font=('Helvetica', 18), bg="white", fg="#4974a5").pack(pady=20)


class ReportsPage(SidebarFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        tk.Label(self, text="Reports", font=('Helvetica', 18), bg="white", fg="#4974a5").pack(pady=20)


class DataAnalysisPage(SidebarFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        tk.Label(self, text="Data Analysis Tools", font=('Helvetica', 18), bg="white", fg="#4974a5").pack(pady=20)


class ModelSelectionPage(SidebarFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        tk.Label(self, text="Model Selection", font=('Helvetica', 18), bg="white", fg="#4974a5").pack(pady=20)


class ModelTrainingPage(SidebarFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        tk.Label(self, text="Model Training", font=('Helvetica', 18), bg="white", fg="#4974a5").pack(pady=20)


class ModelEvaluationPage(SidebarFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        tk.Label(self, text="Model Evaluation", font=('Helvetica', 18), bg="white", fg="#4974a5").pack(pady=20)


if __name__ == "__main__":
    app = AIDSSApp()
    app.mainloop()
