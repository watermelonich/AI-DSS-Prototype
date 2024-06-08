import sys
import os
import tkinter as tk
from tkinter import Canvas, Entry, Button, Scrollbar, filedialog, messagebox
from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
import simpy
import pydotplus
import scipy.stats as stats
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api_wrappers import med, healthsg, fin, newscatcher, supp_chain_opt
from model_wrappers import reg_models_wrapper, clas_models_wrapper

class AIDSSApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI-DSS Sample")
        self.geometry("745x670")
        self.configure(bg="white")

        self.minsize(745, 670)
        self.maxsize(745, 670)

        self.frames = {}
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.create_frames(container)
        self.show_frame("DashboardPage")

    def create_frames(self, container):
        for F in (DashboardPage, FinancePage, HealthPage, MarketingPage, SupplyChainPage, ReportsPage, DataAnalysisPage, ModelSelectionPage, ModelTrainingPage, ModelEvaluationPage, DataManagementPage, ScenarioBuilderPage):
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
            ("Supply Chain", lambda: self.controller.show_frame("SupplyChainPage")),
            ("Data Management", lambda: self.controller.show_frame("DataManagementPage")),
            ("Scenario Builder", lambda: self.controller.show_frame("ScenarioBuilderPage"))
        ]

        for text, command in buttons:
            button = tk.Button(sidebar, text=text, command=command, fg="lightblue", bg='#ECF1F6', bd=0,
                               highlightthickness=0, highlightbackground='#ECF1F6', relief='flat')
            button.pack(fill='x', pady=5)

class ScenarioBuilderPage(SidebarFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        tk.Label(self.scrollable_frame, text="Scenario Builder", font=('Helvetica', 18), bg="white", fg="#4974a5").pack(pady=20)

        tk.Label(self.scrollable_frame, text="Scenario Name", bg="white").pack(pady=5)
        self.scenario_name_entry = tk.Entry(self.scrollable_frame)
        self.scenario_name_entry.pack(pady=5)

        tk.Label(self.scrollable_frame, text="Parameter 1", bg="white").pack(pady=5)
        self.param1_entry = tk.Entry(self.scrollable_frame)
        self.param1_entry.pack(pady=5)

        tk.Label(self.scrollable_frame, text="Parameter 2", bg="white").pack(pady=5)
        self.param2_entry = tk.Entry(self.scrollable_frame)
        self.param2_entry.pack(pady=5)

        self.save_button = tk.Button(self.scrollable_frame, text="Save Scenario", command=self.save_scenario)
        self.save_button.pack(pady=10)

        self.run_monte_carlo_button = tk.Button(self.scrollable_frame, text="Run Monte Carlo Simulation", command=self.run_monte_carlo)
        self.run_monte_carlo_button.pack(pady=10)

        self.sensitivity_analysis_button = tk.Button(self.scrollable_frame, text="Perform Sensitivity Analysis", command=self.perform_sensitivity_analysis)
        self.sensitivity_analysis_button.pack(pady=10)

        self.decision_tree_button = tk.Button(self.scrollable_frame, text="Perform Decision Tree Analysis", command=self.perform_decision_tree_analysis)
        self.decision_tree_button.pack(pady=10)

        self.simulation_button = tk.Button(self.scrollable_frame, text="Run Simulation Model", command=self.run_simulation_model)
        self.simulation_button.pack(pady=10)

        self.stochastic_modeling_button = tk.Button(self.scrollable_frame, text="Perform Stochastic Modeling", command=self.perform_stochastic_modeling)
        self.stochastic_modeling_button.pack(pady=10)

        self.constraint_analysis_button = tk.Button(self.scrollable_frame, text="Perform Constraint Analysis", command=self.perform_constraint_analysis)
        self.constraint_analysis_button.pack(pady=10)

        self.risk_assessment_button = tk.Button(self.scrollable_frame, text="Perform Risk Assessment", command=self.perform_risk_assessment)
        self.risk_assessment_button.pack(pady=10)

        self.scenarios_text = tk.Text(self.scrollable_frame, height=10, wrap='word')
        self.scenarios_text.pack(fill='x', padx=5, pady=5)

    def save_scenario(self):
        scenario_name = self.scenario_name_entry.get()
        param1 = self.param1_entry.get()
        param2 = self.param2_entry.get()

        scenario = f"Scenario: {scenario_name}, Parameter 1: {param1}, Parameter 2: {param2}\n"
        self.scenarios_text.insert(tk.END, scenario)

    def run_monte_carlo(self):
        try:
            param1 = float(self.param1_entry.get())
            param2 = float(self.param2_entry.get())
            num_simulations = 1000  # Number of simulations

            # Assuming normal distribution for simplicity
            param1_simulations = np.random.normal(param1, 1, num_simulations)
            param2_simulations = np.random.normal(param2, 1, num_simulations)

            results = param1_simulations + param2_simulations  # Example result calculation

            figure = plt.Figure(figsize=(5, 4), dpi=100)
            ax = figure.add_subplot(111)
            ax.hist(results, bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax.set_title('Monte Carlo Simulation Results')

            canvas = FigureCanvasTkAgg(figure, self.scrollable_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numerical values for the parameters.")

    def perform_sensitivity_analysis(self):
        try:
            param1 = float(self.param1_entry.get())
            param2 = float(self.param2_entry.get())
            variations = [-0.1, 0, 0.1]  # Example sensitivity variations

            results = []
            for v in variations:
                result = (param1 + param1 * v) + (param2 + param2 * v)  # Example sensitivity result calculation
                results.append(result)

            figure = plt.Figure(figsize=(5, 4), dpi=100)
            ax = figure.add_subplot(111)
            ax.plot(variations, results, marker='o')
            ax.set_title('Sensitivity Analysis')
            ax.set_xlabel('Variation')
            ax.set_ylabel('Result')

            canvas = FigureCanvasTkAgg(figure, self.scrollable_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numerical values for the parameters.")

    def perform_decision_tree_analysis(self):
        try:
            param1 = float(self.param1_entry.get())
            param2 = float(self.param2_entry.get())
            
            # Example data for decision tree analysis
            X = np.array([[param1, param2], [param1+1, param2+1], [param1-1, param2-1]])
            y = np.array([0, 1, 0])  # Example target variable

            clf = DecisionTreeClassifier()
            clf.fit(X, y)

            dot_data = export_graphviz(clf, out_file=None, filled=True, rounded=True, special_characters=True)
            graph = pydotplus.graph_from_dot_data(dot_data)
            plot_path = "decision_tree.png"
            graph.write_png(plot_path)

            # Load the image into Tkinter
            img = Image.open(plot_path)
            img_tk = ImageTk.PhotoImage(img)
            img_label = tk.Label(self.scrollable_frame, image=img_tk)
            img_label.image = img_tk  # Keep a reference to avoid garbage collection
            img_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            # Clean up the plot file
            os.remove(plot_path)
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numerical values for the parameters.")

    def run_simulation_model(self):
        try:
            param1 = float(self.param1_entry.get())
            param2 = float(self.param2_entry.get())
            
            # Custom simulation logic using SimPy
            env = simpy.Environment()
            results = []

            def simple_process(env, param1, param2):
                yield env.timeout(param1)
                results.append(param2)
                yield env.timeout(param2)
                results.append(param1)

            env.process(simple_process(env, param1, param2))
            env.run()

            figure = plt.Figure(figsize=(5, 4), dpi=100)
            ax = figure.add_subplot(111)
            ax.plot(results, marker='o')
            ax.set_title('Simulation Model Results')
            ax.set_xlabel('Step')
            ax.set_ylabel('Value')

            canvas = FigureCanvasTkAgg(figure, self.scrollable_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numerical values for the parameters.")

    def perform_stochastic_modeling(self):
        try:
            param1 = float(self.param1_entry.get())
            param2 = float(self.param2_entry.get())
            num_simulations = 1000

            # Stochastic modeling using normal distribution
            param1_samples = np.random.normal(param1, 1, num_simulations)
            param2_samples = np.random.normal(param2, 1, num_simulations)

            results = param1_samples + param2_samples

            figure = plt.Figure(figsize=(5, 4), dpi=100)
            ax = figure.add_subplot(111)
            sns.histplot(results, bins=50, kde=True, ax=ax)
            ax.set_title('Stochastic Modeling Results')

            canvas = FigureCanvasTkAgg(figure, self.scrollable_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numerical values for the parameters.")

    def perform_constraint_analysis(self):
        try:
            param1 = float(self.param1_entry.get())
            param2 = float(self.param2_entry.get())
            
            # Example constraint analysis
            values = np.array([param1, param2, param1 * 2, param2 * 2])
            constraint = param1 + param2  # Example constraint

            figure = plt.Figure(figsize=(5, 4), dpi=100)
            ax = figure.add_subplot(111)
            ax.plot(values, marker='o', label='Values')
            ax.axhline(y=constraint, color='r', linestyle='--', label='Constraint')
            ax.set_title('Constraint Analysis')
            ax.set_xlabel('Index')
            ax.set_ylabel('Value')
            ax.legend()

            canvas = FigureCanvasTkAgg(figure, self.scrollable_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numerical values for the parameters.")

    def perform_risk_assessment(self):
        try:
            param1 = float(self.param1_entry.get())
            param2 = float(self.param2_entry.get())
            num_simulations = 1000

            # Using normal distributions for risk assessment
            param1_samples = np.random.normal(param1, 1, num_simulations)
            param2_samples = np.random.normal(param2, 1, num_simulations)

            results = param1_samples + param2_samples

            # Risk measure: Value at Risk (VaR) at 95% confidence level
            var_95 = np.percentile(results, 5)

            figure = plt.Figure(figsize=(5, 4), dpi=100)
            ax = figure.add_subplot(111)
            sns.histplot(results, bins=50, kde=True, ax=ax)
            ax.axvline(x=var_95, color='r', linestyle='--', label=f'VaR (95%): {var_95:.2f}')
            ax.set_title('Risk Assessment')
            ax.legend()

            canvas = FigureCanvasTkAgg(figure, self.scrollable_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numerical values for the parameters.")

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

class DataManagementPage(SidebarFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        tk.Label(self.scrollable_frame, text="Data Management", font=('Helvetica', 18), bg="white", fg="#4974a5").pack(pady=20)

        self.upload_button = tk.Button(self.scrollable_frame, text="Upload CSV/Excel", command=self.upload_file)
        self.upload_button.pack(pady=10)

        self.data_text = tk.Text(self.scrollable_frame, height=20, wrap='word')
        self.data_text.pack(fill='x', padx=5, pady=5)

        self.data_cleaning_button = tk.Button(self.scrollable_frame, text="Clean Data", command=self.clean_data)
        self.data_cleaning_button.pack(pady=10)

        self.data_transform_button = tk.Button(self.scrollable_frame, text="Transform Data", command=self.transform_data)
        self.data_transform_button.pack(pady=10)

        self.save_version_button = tk.Button(self.scrollable_frame, text="Save Version", command=self.save_version)
        self.save_version_button.pack(pady=10)

        self.revert_version_button = tk.Button(self.scrollable_frame, text="Revert to Previous Version", command=self.revert_version)
        self.revert_version_button.pack(pady=10)

        self.data = None
        self.data_versions = []
        self.current_version_index = -1

    def save_version(self):
        if self.data is not None:
            self.data_versions.append(self.data.copy())
            self.current_version_index = len(self.data_versions) - 1
            messagebox.showinfo("Version Saved", "Current data version saved.")

    def upload_file(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx'):
                self.data = pd.read_excel(file_path)
            else:
                messagebox.showerror("Invalid file", "Only CSV and Excel files are supported.")
                return
            self.save_version()
            self.display_data(self.data)

    def clean_data(self):
        if self.data is None:
            messagebox.showwarning("No data", "Please upload data first.")
            return

        # Handle missing values
        self.data.fillna(method='ffill', inplace=True)

        # Drop duplicates
        self.data.drop_duplicates(inplace=True)

        # Example outlier handling (removing values beyond 3 standard deviations)
        numeric_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_cols:
            mean = self.data[col].mean()
            std = self.data[col].std()
            self.data = self.data[(self.data[col] > mean - 3 * std) & (self.data[col] < mean + 3 * std)]

        self.save_version()
        self.display_data(self.data)

    def transform_data(self):
        if self.data is None:
            messagebox.showwarning("No data", "Please upload data first.")
            return

        transformation_choice = tk.simpledialog.askstring("Transformation Choice", "Enter 'normalize' for Min-Max Scaling or 'standardize' for Standard Scaling:")

        if transformation_choice == 'normalize':
            # Normalization
            numeric_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
            scaler = MinMaxScaler()
            self.data[numeric_cols] = scaler.fit_transform(self.data[numeric_cols])
        elif transformation_choice == 'standardize':
            # Standardization
            numeric_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
            scaler = StandardScaler()
            self.data[numeric_cols] = scaler.fit_transform(self.data[numeric_cols])
        else:
            messagebox.showerror("Invalid choice", "Invalid choice for transformation. Choose 'normalize' or 'standardize'.")
            return

        # Example encoding categorical variables
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            self.data[col] = LabelEncoder().fit_transform(self.data[col])

        self.save_version()
        self.display_data(self.data)

    def revert_version(self):
        if self.current_version_index > 0:
            self.current_version_index -= 1
            self.data = self.data_versions[self.current_version_index].copy()
            self.display_data(self.data)
            messagebox.showinfo("Version Reverted", "Reverted to previous data version.")
        else:
            messagebox.showwarning("No previous version", "No previous data version available.")

if __name__ == "__main__":
    app = AIDSSApp()
    app.mainloop()
