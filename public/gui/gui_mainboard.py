# BASE


# import tkinter as tk
# from tkinter import ttk
# from PIL import Image, ImageTk
# from api_wrappers import endless_medical, healthsage, finchat, newscatcher, supply_chain_optimizer
# from model_wrappers import reg_models_wrapper, clas_models_wrapper, earth_mapplot_wrapper

# class AIDSSApp(tk.Tk):
#     def __init__(self):
#         super().__init__()
#         self.title("AI-DSS Sample")
#         self.geometry("880x610")

#         # Create sidebar and main content frame
#         self.sidebar = tk.Frame(self, width=200, bg='lightgrey')
#         self.sidebar.pack(expand=False, fill='both', side='left', anchor='nw')

#         self.main_content = tk.Frame(self, bg='white')
#         self.main_content.pack(expand=True, fill='both', side='right')

#         self.create_sidebar()
#         self.create_home_panel()

#     def create_sidebar(self):
#         tk.Label(self.sidebar, text="Categories", bg='lightgrey', font=('Helvetica', 16, 'bold')).pack(pady=10)

#         tk.Button(self.sidebar, text="Home", command=self.create_home_panel).pack(fill='x')
#         tk.Button(self.sidebar, text="Finance", command=self.create_finance_panel).pack(fill='x')
#         tk.Button(self.sidebar, text="Health/Med", command=self.create_health_panel).pack(fill='x')
#         tk.Button(self.sidebar, text="Marketing", command=self.create_marketing_panel).pack(fill='x')
#         tk.Button(self.sidebar, text="Supply Chain", command=self.create_supply_chain_panel).pack(fill='x')

#     def clear_main_content(self):
#         for widget in self.main_content.winfo_children():
#             widget.destroy()

#     def create_home_panel(self):
#         self.clear_main_content()
#         tk.Label(self.main_content, text="Welcome to the AI-DSS Sample Application", font=('Helvetica', 18)).pack(pady=20)

#     def create_finance_panel(self):
#         self.clear_main_content()
#         self.create_section("Finchat Data", self.fetch_finchat_data)

#     def create_health_panel(self):
#         self.clear_main_content()
#         self.create_section("Endless Medical Data", self.fetch_medical_data)
#         self.create_section("Healthsage Data", self.fetch_healthsage_data)
        
#     def create_marketing_panel(self):
#         self.clear_main_content()
#         self.create_plot_section("Logistic Regression", self.plot_logistic_regression)

#     def create_supply_chain_panel(self):
#         self.clear_main_content()
#         self.create_section("Supply Chain Data", self.fetch_supply_chain_data)
#         self.create_plot_section("Earth Map Plot", self.plot_earth_map)
#         self.create_plot_section("Decision Tree Regressor", self.plot_decision_tree_regressor)

#     def create_section(self, title, command):
#         frame = tk.Frame(self.main_content, pady=10)
#         frame.pack(fill='x')
#         label = tk.Label(frame, text=title, font=('Helvetica', 14))
#         label.pack(side='left')
#         input_entry = tk.Entry(frame)
#         input_entry.pack(side='left', padx=5)
#         button = tk.Button(frame, text=f"Get {title.split()[0]} Data", command=lambda: command(input_entry))
#         button.pack(side='left', padx=5)
#         result_text = tk.Text(self.main_content, height=5, wrap='word')
#         result_text.pack(fill='x', padx=5, pady=5)
#         setattr(self, f"{title.split()[0].lower()}_result", result_text)

#     def create_plot_section(self, title, command):
#         frame = tk.Frame(self.main_content, pady=10)
#         frame.pack(fill='x')
#         label = tk.Label(frame, text=title, font=('Helvetica', 14))
#         label.pack(side='left')
#         button = tk.Button(frame, text=f"Plot {title}", command=command)
#         button.pack(side='left', padx=5)
#         canvas = tk.Label(self.main_content)
#         canvas.pack(fill='x', padx=5, pady=5)
#         setattr(self, f"{title.replace(' ', '_').lower()}_canvas", canvas)

#     def fetch_medical_data(self, entry):
#         params = {"param": entry.get()}
#         data = endless_medical.get_medical_data(params)
#         self.medical_result.delete(1.0, tk.END)
#         self.medical_result.insert(tk.END, str(data))

#     def fetch_healthsage_data(self, entry):
#         params = {"param": entry.get()}
#         data = healthsage.get_healthsage_data(params)
#         self.healthsage_result.delete(1.0, tk.END)
#         self.healthsage_result.insert(tk.END, str(data))

#     def fetch_finchat_data(self, entry):
#         params = {"param": entry.get()}
#         data = finchat.get_financial_data(params)
#         self.finchat_result.delete(1.0, tk.END)
#         self.finchat_result.insert(tk.END, str(data))

#     def fetch_newscatcher_data(self, entry):
#         params = {"param": entry.get()}
#         data = newscatcher.get_news_data(params)
#         self.newscatcher_result.delete(1.0, tk.END)
#         self.newscatcher_result.insert(tk.END, str(data))

#     def fetch_supply_chain_data(self, entry):
#         params = {"param": entry.get()}
#         data = supply_chain_optimizer.get_supply_chain_data(params)
#         self.supply_chain_result.delete(1.0, tk.END)
#         self.supply_chain_result.insert(tk.END, str(data))

#     def plot_decision_tree_regressor(self):
#         plot_path = reg_models_wrapper.run_decision_tree_regressor()
#         self.display_plot(self.decision_tree_regressor_canvas, plot_path)

#     def plot_logistic_regression(self):
#         plot_path = clas_models_wrapper.run_logistic_regression()
#         self.display_plot(self.logistic_regression_canvas, plot_path)

#     def plot_earth_map(self):
#         plot_path = earth_mapplot_wrapper.generate_earth_map_plot()
#         self.display_plot(self.earth_map_plot_canvas, plot_path)

#     def display_plot(self, canvas, plot_path):
#         image = Image.open(plot_path)
#         photo = ImageTk.PhotoImage(image)
#         canvas.config(image=photo)
#         canvas.image = photo

# if __name__ == "__main__":
#     app = AIDSSApp()
#     app.mainloop()





# USING FRAMES, TO TEST


# import tkinter as tk
# from tkinter import ttk
# from PIL import Image, ImageTk
# from api_wrappers import endless_medical, healthsage, finchat, newscatcher, supply_chain_optimizer
# from model_wrappers import reg_models_wrapper, clas_models_wrapper, earth_mapplot_wrapper

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from api_wrappers import med, healthsg, fin, newscatcher, supp_chain_opt
from model_wrappers import reg_models_wrapper, clas_models_wrapper, earth_mapplot_wrapper


class AIDSSApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI-DSS Sample")
        self.geometry("880x610")

        self.frames = {}
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.create_frames(container)
        self.show_frame("HomePage")

    def create_frames(self, container):
        self.frames["HomePage"] = HomePage(parent=container, controller=self)
        self.frames["FinancePage"] = FinancePage(parent=container, controller=self)
        self.frames["HealthPage"] = HealthPage(parent=container, controller=self)
        self.frames["MarketingPage"] = MarketingPage(parent=container, controller=self)
        self.frames["SupplyChainPage"] = SupplyChainPage(parent=container, controller=self)

        for frame in self.frames.values():
            frame.grid(row=0, column=0, sticky="nsew")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()

class HomePage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        tk.Label(self, text="Welcome to the AI-DSS Sample Application", font=('Helvetica', 18)).pack(pady=20)
        self.create_sidebar()

    def create_sidebar(self):
        sidebar = tk.Frame(self, width=200, bg='lightgrey')
        sidebar.pack(expand=False, fill='both', side='left', anchor='nw')
        tk.Label(sidebar, text="Categories", bg='lightgrey', font=('Helvetica', 16, 'bold')).pack(pady=10)

        tk.Button(sidebar, text="Home", command=lambda: self.controller.show_frame("HomePage")).pack(fill='x')
        tk.Button(sidebar, text="Finance", command=lambda: self.controller.show_frame("FinancePage")).pack(fill='x')
        tk.Button(sidebar, text="Health/Med", command=lambda: self.controller.show_frame("HealthPage")).pack(fill='x')
        tk.Button(sidebar, text="Marketing", command=lambda: self.controller.show_frame("MarketingPage")).pack(fill='x')
        tk.Button(sidebar, text="Supply Chain", command=lambda: self.controller.show_frame("SupplyChainPage")).pack(fill='x')

class FinancePage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.create_sidebar()
        self.create_finance_panel()

    def create_sidebar(self):
        sidebar = tk.Frame(self, width=200, bg='lightgrey')
        sidebar.pack(expand=False, fill='both', side='left', anchor='nw')
        tk.Label(sidebar, text="Categories", bg='lightgrey', font=('Helvetica', 16, 'bold')).pack(pady=10)

        tk.Button(sidebar, text="Home", command=lambda: self.controller.show_frame("HomePage")).pack(fill='x')
        tk.Button(sidebar, text="Finance", command=lambda: self.controller.show_frame("FinancePage")).pack(fill='x')
        tk.Button(sidebar, text="Health/Med", command=lambda: self.controller.show_frame("HealthPage")).pack(fill='x')
        tk.Button(sidebar, text="Marketing", command=lambda: self.controller.show_frame("MarketingPage")).pack(fill='x')
        tk.Button(sidebar, text="Supply Chain", command=lambda: self.controller.show_frame("SupplyChainPage")).pack(fill='x')

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
        data = finchat.get_financial_data(params)
        self.finchat_result.delete(1.0, tk.END)
        self.finchat_result.insert(tk.END, str(data))

class HealthPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.create_sidebar()
        self.create_health_panel()

    def create_sidebar(self):
        sidebar = tk.Frame(self, width=200, bg='lightgrey')
        sidebar.pack(expand=False, fill='both', side='left', anchor='nw')
        tk.Label(sidebar, text="Categories", bg='lightgrey', font=('Helvetica', 16, 'bold')).pack(pady=10)

        tk.Button(sidebar, text="Home", command=lambda: self.controller.show_frame("HomePage")).pack(fill='x')
        tk.Button(sidebar, text="Finance", command=lambda: self.controller.show_frame("FinancePage")).pack(fill='x')
        tk.Button(sidebar, text="Health/Med", command=lambda: self.controller.show_frame("HealthPage")).pack(fill='x')
        tk.Button(sidebar, text="Marketing", command=lambda: self.controller.show_frame("MarketingPage")).pack(fill='x')
        tk.Button(sidebar, text="Supply Chain", command=lambda: self.controller.show_frame("SupplyChainPage")).pack(fill='x')

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
        data = endless_medical.get_medical_data(params)
        self.medical_result.delete(1.0, tk.END)
        self.medical_result.insert(tk.END, str(data))

    def fetch_healthsage_data(self, entry):
        params = {"param": entry.get()}
        data = healthsage.get_healthsage_data(params)
        self.healthsage_result.delete(1.0, tk.END)
        self.healthsage_result.insert(tk.END, str(data))

class MarketingPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.create_sidebar()
        self.create_marketing_panel()

    def create_sidebar(self):
        sidebar = tk.Frame(self, width=200, bg='lightgrey')
        sidebar.pack(expand=False, fill='both', side='left', anchor='nw')
        tk.Label(sidebar, text="Categories", bg='lightgrey', font=('Helvetica', 16, 'bold')).pack(pady=10)

        tk.Button(sidebar, text="Home", command=lambda: self.controller.show_frame("HomePage")).pack(fill='x')
        tk.Button(sidebar, text="Finance", command=lambda: self.controller.show_frame("FinancePage")).pack(fill='x')
        tk.Button(sidebar, text="Health/Med", command=lambda: self.controller.show_frame("HealthPage")).pack(fill='x')
        tk.Button(sidebar, text="Marketing", command=lambda: self.controller.show_frame("MarketingPage")).pack(fill='x')
        tk.Button(sidebar, text="Supply Chain", command=lambda: self.controller.show_frame("SupplyChainPage")).pack(fill='x')

    def create_marketing_panel(self):
        frame = tk.Frame(self, pady=10)
        frame.pack(fill='x')
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

class SupplyChainPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.create_sidebar()
        self.create_supply_chain_panel()

    def create_sidebar(self):
        sidebar = tk.Frame(self, width=200, bg='lightgrey')
        sidebar.pack(expand=False, fill='both', side='left', anchor='nw')
        tk.Label(sidebar, text="Categories", bg='lightgrey', font=('Helvetica', 16, 'bold')).pack(pady=10)

        tk.Button(sidebar, text="Home", command=lambda: self.controller.show_frame("HomePage")).pack(fill='x')
        tk.Button(sidebar, text="Finance", command=lambda: self.controller.show_frame("FinancePage")).pack(fill='x')
        tk.Button(sidebar, text="Health/Med", command=lambda: self.controller.show_frame("HealthPage")).pack(fill='x')
        tk.Button(sidebar, text="Marketing", command=lambda: self.controller.show_frame("MarketingPage")).pack(fill='x')
        tk.Button(sidebar, text="Supply Chain", command=lambda: self.controller.show_frame("SupplyChainPage")).pack(fill='x')

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
        data = supply_chain_optimizer.get_supply_chain_data(params)
        self.supply_chain_result.delete(1.0, tk.END)
        self.supply_chain_result.insert(tk.END, str(data))

if __name__ == "__main__":
    app = AIDSSApp()
    app.mainloop()
