import sys
import os
import tkinter as tk
from tkinter import Canvas, Entry, Button, Scrollbar, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
import simpy
import pydotplus
import scipy.stats as stats
import seaborn as sns
import requests
from requests.exceptions import HTTPError
import logging
from confluent_kafka import Producer, Consumer, KafkaException
from tkinter import ttk

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api_wrappers import med, healthsg, fin, newscatcher, supp_chain_opt
from model_wrappers import reg_models_wrapper, clas_models_wrapper

# Set up logging for API monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        for F in (DashboardPage, FinancePage, HealthPage, MarketingPage, SupplyChainPage, ReportsPage, DataAnalysisPage, ModelSelectionPage, ModelTrainingPage, ModelEvaluationPage, DataManagementPage, ScenarioBuilderPage, APIIntegrationPage):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()

class UserSetupPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.configure(bg="white")
        
        tk.Label(self, text="User Setup", font=('Helvetica', 18), bg="white", fg="#4974a5").pack(pady=20)

        tk.Label(self, text="Role:", bg="white").pack(pady=5)
        self.role_entry = tk.Entry(self)
        self.role_entry.pack(pady=5)

        tk.Label(self, text="User Information:", bg="white").pack(pady=5)
        self.user_info_entry = tk.Entry(self)
        self.user_info_entry.pack(pady=5)

        tk.Label(self, text="Company Type:", bg="white").pack(pady=5)
        self.company_type_entry = tk.Entry(self)
        self.company_type_entry.pack(pady=5)

        tk.Label(self, text="Sector:", bg="white").pack(pady=5)

        sectors = [("Finance", "finance.png"), ("Marketing", "marketing.png"), ("Supply Chain", "supply_chain.png"), ("Health/Med", "health_med.png")]
        self.selected_sector = tk.StringVar()
        
        for sector, icon in sectors:
            frame = tk.Frame(self, bg="white")
            frame.pack(pady=5)
            icon_img = tk.PhotoImage(file=f"../assets/{icon}")
            tk.Radiobutton(frame, text=sector, variable=self.selected_sector, value=sector, image=icon_img, compound='left', bg="white", indicatoron=0).pack(side='left')

        tk.Button(self, text="Submit", command=self.submit).pack(pady=20)

    def submit(self):
        role = self.role_entry.get()
        user_info = self.user_info_entry.get()
        company_type = self.company_type_entry.get()
        sector = self.selected_sector.get()

        # You can store this information as needed and then navigate to the selected sector page
        print(f"Role: {role}, User Info: {user_info}, Company Type: {company_type}, Sector: {sector}")
        self.controller.show_frame(f"{sector}Page")

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
            ("Scenario Builder", lambda: self.controller.show_frame("ScenarioBuilderPage")),
            ("API Integration", lambda: self.controller.show_frame("APIIntegrationPage"))
        ]

        for text, command in buttons:
            button = tk.Button(sidebar, text=text, command=command, fg="lightblue", bg='#ECF1F6', bd=0,
                               highlightthickness=0, highlightbackground='#ECF1F6', relief='flat')
            button.pack(fill='x', pady=5)

class APIIntegrationPage(SidebarFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        tk.Label(self.scrollable_frame, text="API Integration", font=('Helvetica', 18), bg="white", fg="#4974a5").pack(pady=20)

        self.api_url_entry = tk.Entry(self.scrollable_frame)
        self.api_url_entry.pack(pady=5)
        self.api_key_entry = tk.Entry(self.scrollable_frame)
        self.api_key_entry.pack(pady=5)
        self.api_data_text = tk.Text(self.scrollable_frame, height=10, wrap='word')
        self.api_data_text.pack(fill='x', padx=5, pady=5)

        self.api_connect_button = tk.Button(self.scrollable_frame, text="Connect to API", command=self.connect_to_api)
        self.api_connect_button.pack(pady=10)

        self.kafka_producer_button = tk.Button(self.scrollable_frame, text="Start Kafka Producer", command=self.start_kafka_producer)
        self.kafka_producer_button.pack(pady=10)

        self.kafka_consumer_button = tk.Button(self.scrollable_frame, text="Start Kafka Consumer", command=self.start_kafka_consumer)
        self.kafka_consumer_button.pack(pady=10)

    def connect_to_api(self):
        url = self.api_url_entry.get()
        headers = {'Authorization': f'Bearer {self.api_key_entry.get()}'}
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            self.api_data_text.delete(1.0, tk.END)
            self.api_data_text.insert(tk.END, response.json())
        except HTTPError as http_err:
            logging.error(f"HTTP error occurred: {http_err}")
            messagebox.showerror("HTTP Error", f"HTTP error occurred: {http_err}")
        except Exception as err:
            logging.error(f"Other error occurred: {err}")
            messagebox.showerror("Error", f"Other error occurred: {err}")

    def start_kafka_producer(self):
        conf = {'bootstrap.servers': "localhost:9092"}
        producer = Producer(**conf)
        
        def acked(err, msg):
            if err is not None:
                logging.error(f"Failed to deliver message: {err}")
            else:
                logging.info(f"Message produced: {msg.topic()} [{msg.partition()}] @ {msg.offset()}")

        def send_messages():
            for _ in range(100):
                producer.produce('test-topic', key="key", value="value", callback=acked)
            producer.flush()
            messagebox.showinfo("Kafka Producer", "Messages sent to Kafka topic.")
        
        send_messages()

    def start_kafka_consumer(self):
        conf = {
            'bootstrap.servers': "localhost:9092",
            'group.id': "group1",
            'auto.offset.reset': 'earliest'
        }
        consumer = Consumer(**conf)
        consumer.subscribe(['test-topic'])

        def consume_messages():
            try:
                while True:
                    msg = consumer.poll(timeout=1.0)
                    if msg is None:
                        continue
                    if msg.error():
                        if msg.error().code() == KafkaError._PARTITION_EOF:
                            logging.info(f"End of partition reached {msg.topic()}/{msg.partition()}")
                        elif msg.error():
                            raise KafkaException(msg.error())
                    else:
                        self.api_data_text.insert(tk.END, f"{msg.value().decode('utf-8')}\n")
            except KafkaException as e:
                logging.error(f"Kafka exception: {e}")
            finally:
                consumer.close()
        
        consume_messages()

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

class ConfigDialog(simpledialog.Dialog):
    def body(self, master):
        self.title("Dashboard Configuration")
        
        self.config = {}
        
        sections = ["Finance", "Health", "Marketing", "Supply Chain"]
        self.options = ["Line Plot", "Bar Plot", "Pie Chart", "Scatter Plot"]
        
        for i, section in enumerate(sections):
            tk.Label(master, text=section).grid(row=i, column=0)
            self.config[section] = tk.StringVar()
            combo = ttk.Combobox(master, textvariable=self.config[section], values=self.options, state='readonly')
            combo.grid(row=i, column=1)
            combo.current(0)
        
    def apply(self):
        self.result = {section: self.config[section].get() for section in self.config}

class SidebarFrame(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.scrollable_frame = tk.Frame(self, bg="white")
        self.scrollable_frame.pack(fill='both', expand=True)
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
            ("Scenario Builder", lambda: self.controller.show_frame("ScenarioBuilderPage")),
            ("API Integration", lambda: self.controller.show_frame("APIIntegrationPage"))
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

        self.config_button = tk.Button(self.scrollable_frame, text="Configure Dashboard", command=self.open_config_dialog, bg='white', fg='#4974a5')
        self.config_button.pack(pady=10)

        self.config = None
        self.update_dashboard()

    def open_config_dialog(self):
        dialog = ConfigDialog(self)
        if dialog.result:
            self.config = dialog.result
            self.update_dashboard()

    def update_dashboard(self):
        if self.config is None:
            self.config = {"Finance": "Line Plot", "Health": "Line Plot", "Marketing": "Line Plot", "Supply Chain": "Line Plot"}

        finance_data = self.get_finance_data_summary()
        health_data = self.get_health_data_summary()
        marketing_data = self.get_marketing_data_summary()
        supply_chain_data = self.get_supply_chain_data_summary()

        self.create_plot("Finance Summary", finance_data, self.config["Finance"])
        self.create_plot("Health Summary", health_data, self.config["Health"])
        self.create_plot("Marketing Summary", marketing_data, self.config["Marketing"])
        self.create_plot("Supply Chain Summary", supply_chain_data, self.config["Supply Chain"])
        self.create_sector_pie_chart(finance_data, health_data, marketing_data, supply_chain_data)

    def create_plot(self, title, data, plot_type):
        frame = tk.Frame(self.scrollable_frame, bg="white")
        frame.pack(fill='both', expand=True, pady=10)
        label = tk.Label(frame, text=title, font=('Helvetica', 12), bg="white")
        label.pack()
        figure = Figure(figsize=(5, 4), dpi=100)
        ax = figure.add_subplot(111)
        if plot_type == "Line Plot":
            for series in data:
                ax.plot(series['x'], series['y'], label=series['label'])
        elif plot_type == "Bar Plot":
            for series in data:
                ax.bar(series['x'], series['y'], label=series['label'])
        elif plot_type == "Pie Chart":
            if data:
                series = data[0]
                ax.pie(series['y'], labels=series['x'], autopct='%1.1f%%', startangle=140)
        elif plot_type == "Scatter Plot":
            for series in data:
                ax.scatter(series['x'], series['y'], label=series['label'])
        ax.set_title(title)
        ax.legend()
        canvas = FigureCanvasTkAgg(figure, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def create_sector_pie_chart(self, finance_data, health_data, marketing_data, supply_chain_data):
        frame = tk.Frame(self.scrollable_frame, bg="white")
        frame.pack(fill='both', expand=True, pady=10)
        label = tk.Label(frame, text="Sector Distribution", font=('Helvetica', 12), bg="white")
        label.pack()
        figure = Figure(figsize=(5, 4), dpi=100)
        ax = figure.add_subplot(111)
        sectors = ['Finance', 'Health', 'Marketing', 'Supply Chain']
        values = [len(finance_data), len(health_data), len(marketing_data), len(supply_chain_data)]
        ax.pie(values, labels=sectors, autopct='%1.1f%%', startangle=140)
        ax.set_title('Sector Distribution')
        canvas = FigureCanvasTkAgg(figure, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def get_finance_data_summary(self):
        return [
            {'x': ['2023-01-01', '2023-02-01'], 'y': [100, 110], 'label': 'Stock Prices'},
            {'x': ['2023-01-01', '2023-02-01'], 'y': [200, 190], 'label': 'Economic Indicators'}
        ]

    def get_health_data_summary(self):
        return [
            {'x': ['2023-01-01', '2023-02-01'], 'y': [70, 65], 'label': 'Patient Data'},
            {'x': ['2023-01-01', '2023-02-01'], 'y': [90, 85], 'label': 'Health Metrics'}
        ]

    def get_marketing_data_summary(self):
        return [
            {'x': ['2023-01-01', '2023-02-01'], 'y': [50, 55], 'label': 'Campaign Performance'},
            {'x': ['2023-01-01', '2023-02-01'], 'y': [150, 145], 'label': 'Social Media Metrics'}
        ]

    def get_supply_chain_data_summary(self):
        return [
            {'x': ['2023-01-01', '2023-02-01'], 'y': [400, 390], 'label': 'Inventory Levels'},
            {'x': ['2023-01-01', '2023-02-01'], 'y': [30, 25], 'label': 'Delivery Times'}
        ]

    # Mock implementations for fetching data
    fin = newscatcher = med = healthsg = supp_chain_opt = type('mock', (), {
        'get_financial_data': lambda params: {'stock_prices': {'date': ['2023-01-01', '2023-02-01'], 'value': [100, 110]}, 'economic_indicators': {'date': ['2023-01-01', '2023-02-01'], 'value': [200, 190]}},
        'get_news_data': lambda params: {'campaign_performance': {'date': ['2023-01-01', '2023-02-01'], 'value': [50, 55]}, 'social_media_metrics': {'date': ['2023-01-01', '2023-02-01'], 'value': [150, 145]}},
        'get_medical_data': lambda params: {'patient_data': {'date': ['2023-01-01', '2023-02-01'], 'value': [70, 65]}, 'health_metrics': {'date': ['2023-01-01', '2023-02-01'], 'value': [90, 85]}},
        'get_healthsage_data': lambda params: {'patient_data': {'date': ['2023-01-01', '2023-02-01'], 'value': [70, 65]}, 'health_metrics': {'date': ['2023-01-01', '2023-02-01'], 'value': [90, 85]}},
        'get_supply_chain_data': lambda params: {'inventory_levels': {'date': ['2023-01-01', '2023-02-01'], 'value': [400, 390]}, 'delivery_times': {'date': ['2023-01-01', '2023-02-01'], 'value': [30, 25]}}
    })()


# ------------------------------------- # ------------------------------------- # ------------------------------------- 


class DataInputDialog(simpledialog.Dialog):
    def body(self, master):
        self.title("Load Dataset")
        tk.Label(master, text="Variables:").grid(row=0)
        tk.Label(master, text="Data Type:").grid(row=1)
        tk.Label(master, text="Charts:").grid(row=2)
        tk.Label(master, text="Models:").grid(row=3)

        self.variables_entry = tk.Entry(master)
        self.data_type_entry = tk.Entry(master)
        self.charts_entry = tk.Entry(master)
        self.models_entry = tk.Entry(master)

        self.variables_entry.grid(row=0, column=1)
        self.data_type_entry.grid(row=1, column=1)
        self.charts_entry.grid(row=2, column=1)
        self.models_entry.grid(row=3, column=1)

    def apply(self):
        self.variables = self.variables_entry.get()
        self.data_type = self.data_type_entry.get()
        self.charts = self.charts_entry.get()
        self.models = self.models_entry.get()

class SidebarFrame(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.scrollable_frame = tk.Frame(self, bg="white")
        self.scrollable_frame.pack(fill='both', expand=True)
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
            ("Scenario Builder", lambda: self.controller.show_frame("ScenarioBuilderPage")),
            ("API Integration", lambda: self.controller.show_frame("APIIntegrationPage"))
        ]

        for text, command in buttons:
            button = tk.Button(sidebar, text=text, command=command, fg="lightblue", bg='#ECF1F6', bd=0,
                               highlightthickness=0, highlightbackground='#ECF1F6', relief='flat')
            button.pack(fill='x', pady=5)

class FinancePage(SidebarFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.create_finance_panel()

    def create_finance_panel(self):
        finance_frame = tk.Frame(self.scrollable_frame, pady=10, bg="white")
        finance_frame.pack(fill='x')
        finance_label = tk.Label(finance_frame, text="Finance Data", font=('Helvetica', 16, 'bold'), bg="white")
        finance_label.pack(side='left')
        self.finance_input_entry = tk.Entry(finance_frame)
        self.finance_input_entry.pack(side='left', padx=10)
        finance_get_data_button = tk.Button(finance_frame, text="Get Data", command=self.fetch_finance_data, bg='white', fg='#4974a5')
        finance_get_data_button.pack(side='left', padx=10)
        finance_load_dataset_button = tk.Button(finance_frame, text="Load Dataset", command=self.load_finance_dataset, bg='white', fg='#4974a5')
        finance_load_dataset_button.pack(side='left', padx=10)
        
        self.finance_result_text = tk.Text(self.scrollable_frame, height=5, wrap='word', bg='#ECF1F6')
        self.finance_result_text.pack(fill='x', padx=10, pady=10)

        trend_insights_frame = tk.LabelFrame(self.scrollable_frame, text="Trend Insights", bg="white", padx=10, pady=10)
        trend_insights_frame.pack(fill='both', expand=True, padx=10, pady=10)
        self.finance_trend_figure_frame = tk.Frame(trend_insights_frame, bg="white")
        self.finance_trend_figure_frame.pack(fill='both', expand=True, pady=10)

        business_data_insights_frame = tk.LabelFrame(self.scrollable_frame, text="Business Data Insights", bg="white", padx=10, pady=10)
        business_data_insights_frame.pack(fill='both', expand=True, padx=10, pady=10)
        self.finance_business_figure_frame = tk.Frame(business_data_insights_frame, bg="white")
        self.finance_business_figure_frame.pack(fill='both', expand=True, pady=10)

        # Add empty plots
        self.add_empty_plots(self.finance_trend_figure_frame, "Finance Trend Plot")
        self.add_empty_plots(self.finance_trend_figure_frame, "Finance Trend Plot")
        self.add_empty_plots(self.finance_business_figure_frame, "Finance Business Plot")
        self.add_empty_plots(self.finance_business_figure_frame, "Finance Business Plot")

    def fetch_finance_data(self):
        params = {"param": self.finance_input_entry.get()}
        try:
            data = fin.get_financial_data(params)
            self.finance_result_text.delete(1.0, tk.END)
            self.finance_result_text.insert(tk.END, str(data))
            self.update_finance_trend_plot(data)
            self.update_finance_business_plot(data)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to fetch finance data: {e}")

    def load_finance_dataset(self):
        dialog = DataInputDialog(self)
        if dialog.variables and dialog.data_type and dialog.charts and dialog.models:
            data = {"variables": dialog.variables, "data_type": dialog.data_type, "charts": dialog.charts, "models": dialog.models}
            self.plot_finance_custom_data(data)

    def plot_finance_custom_data(self, data):
        # Implementation of data plotting based on the collected user input
        pass

    def update_finance_trend_plot(self, data):
        for widget in self.finance_trend_figure_frame.winfo_children():
            widget.destroy()
        
        finance_figure = Figure(figsize=(10, 4), dpi=100)
        finance_ax1 = finance_figure.add_subplot(121)
        finance_ax2 = finance_figure.add_subplot(122)

        if 'stock_prices' in data:
            dates = [pd.to_datetime(d) for d in data['stock_prices']['date']]
            values = data['stock_prices']['value']
            finance_ax1.plot(dates, values, marker='o', linestyle='-', color='b', label='Stock Prices')
            finance_ax1.set_title('Stock Prices')
        
        if 'economic_indicators' in data:
            dates = [pd.to_datetime(d) for d in data['economic_indicators']['date']]
            values = data['economic_indicators']['value']
            finance_ax2.plot(dates, values, marker='x', linestyle='--', color='r', label='Economic Indicators')
            finance_ax2.set_title('Economic Indicators')

        finance_ax1.set_xlabel('Date')
        finance_ax1.set_ylabel('Value')
        finance_ax1.legend()

        finance_ax2.set_xlabel('Date')
        finance_ax2.set_ylabel('Value')
        finance_ax2.legend()

        finance_canvas = FigureCanvasTkAgg(finance_figure, self.finance_trend_figure_frame)
        finance_canvas.draw()
        finance_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def update_finance_business_plot(self, data):
        for widget in self.finance_business_figure_frame.winfo_children():
            widget.destroy()
        
        finance_figure = Figure(figsize=(10, 4), dpi=100)
        finance_ax3 = finance_figure.add_subplot(121)
        finance_ax4 = finance_figure.add_subplot(122)

        if 'revenue' in data:
            dates = [pd.to_datetime(d) for d in data['revenue']['date']]
            values = data['revenue']['value']
            finance_ax3.bar(dates, values, color='g', label='Revenue')
            finance_ax3.set_title('Revenue')

        if 'profit_margins' in data:
            labels = data['profit_margins']['labels']
            sizes = data['profit_margins']['sizes']
            finance_ax4.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
            finance_ax4.set_title('Profit Margins')

        finance_ax3.set_xlabel('Date')
        finance_ax3.set_ylabel('Value')
        finance_ax3.legend()

        finance_ax4.set_xlabel('Date')
        finance_ax4.set_ylabel('Value')
        finance_ax4.legend()

        finance_canvas = FigureCanvasTkAgg(finance_figure, self.finance_business_figure_frame)
        finance_canvas.draw()
        finance_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def add_empty_plots(self, frame, title):
        figure = Figure(figsize=(10, 4), dpi=100)
        ax1 = figure.add_subplot(121)
        ax2 = figure.add_subplot(122)
        ax1.set_title(f'{title} 1')
        ax2.set_title(f'{title} 2')
        canvas = FigureCanvasTkAgg(figure, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

class MarketingPage(SidebarFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.create_marketing_panel()

    def create_marketing_panel(self):
        marketing_frame = tk.Frame(self.scrollable_frame, pady=10, bg="white")
        marketing_frame.pack(fill='x')
        marketing_label = tk.Label(marketing_frame, text="Marketing Data", font=('Helvetica', 16, 'bold'), bg="white")
        marketing_label.pack(side='left')
        self.marketing_input_entry = tk.Entry(marketing_frame)
        self.marketing_input_entry.pack(side='left', padx=10)
        marketing_get_data_button = tk.Button(marketing_frame, text="Get Data", command=self.fetch_newscatcher_data, bg='white', fg='#4974a5')
        marketing_get_data_button.pack(side='left', padx=10)
        marketing_load_dataset_button = tk.Button(marketing_frame, text="Load Dataset", command=self.load_marketing_dataset, bg='white', fg='#4974a5')
        marketing_load_dataset_button.pack(side='left', padx=10)
        
        self.marketing_result_text = tk.Text(self.scrollable_frame, height=10, wrap='word', bg='#ECF1F6')
        self.marketing_result_text.pack(fill='x', padx=10, pady=10)

        trend_insights_frame = tk.LabelFrame(self.scrollable_frame, text="Trend Insights", bg="white", padx=10, pady=10)
        trend_insights_frame.pack(fill='both', expand=True, padx=10, pady=10)
        self.marketing_trend_figure_frame = tk.Frame(trend_insights_frame, bg="white")
        self.marketing_trend_figure_frame.pack(fill='both', expand=True, pady=10)

        business_data_insights_frame = tk.LabelFrame(self.scrollable_frame, text="Business Data Insights", bg="white", padx=10, pady=10)
        business_data_insights_frame.pack(fill='both', expand=True, padx=10, pady=10)
        self.marketing_business_figure_frame = tk.Frame(business_data_insights_frame, bg="white")
        self.marketing_business_figure_frame.pack(fill='both', expand=True, pady=10)

        # Add empty plots
        self.add_empty_plots(self.marketing_trend_figure_frame, "Marketing Trend Plot")
        self.add_empty_plots(self.marketing_trend_figure_frame, "Marketing Trend Plot")
        self.add_empty_plots(self.marketing_business_figure_frame, "Marketing Business Plot")
        self.add_empty_plots(self.marketing_business_figure_frame, "Marketing Business Plot")

    def fetch_newscatcher_data(self):
        params = {"param": self.marketing_input_entry.get()}
        try:
            data = newscatcher.get_news_data(params)
            self.marketing_result_text.delete(1.0, tk.END)
            self.marketing_result_text.insert(tk.END, str(data))
            self.update_marketing_trend_plot(data)
            self.update_marketing_business_plot(data)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to fetch marketing data: {e}")

    def load_marketing_dataset(self):
        dialog = DataInputDialog(self)
        if dialog.variables and dialog.data_type and dialog.charts and dialog.models:
            data = {"variables": dialog.variables, "data_type": dialog.data_type, "charts": dialog.charts, "models": dialog.models}
            self.plot_marketing_custom_data(data)

    def plot_marketing_custom_data(self, data):
        pass

    def update_marketing_trend_plot(self, data):
        for widget in self.marketing_trend_figure_frame.winfo_children():
            widget.destroy()
        
        marketing_figure = Figure(figsize=(10, 4), dpi=100)
        marketing_ax1 = marketing_figure.add_subplot(121)
        marketing_ax2 = marketing_figure.add_subplot(122)
        
        if 'campaign_performance' in data:
            dates = [pd.to_datetime(d) for d in data['campaign_performance']['date']]
            values = data['campaign_performance']['value']
            marketing_ax1.plot(dates, values, marker='o', linestyle='-', color='b', label='Campaign Performance')
            marketing_ax1.set_title('Campaign Performance')
        
        if 'social_media_metrics' in data:
            dates = [pd.to_datetime(d) for d in data['social_media_metrics']['date']]
            values = data['social_media_metrics']['value']
            marketing_ax2.plot(dates, values, marker='x', linestyle='--', color='r', label='Social Media Metrics')
            marketing_ax2.set_title('Social Media Metrics')

        marketing_ax1.set_xlabel('Date')
        marketing_ax1.set_ylabel('Value')
        marketing_ax1.legend()

        marketing_ax2.set_xlabel('Date')
        marketing_ax2.set_ylabel('Value')
        marketing_ax2.legend()

        marketing_canvas = FigureCanvasTkAgg(marketing_figure, self.marketing_trend_figure_frame)
        marketing_canvas.draw()
        marketing_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def update_marketing_business_plot(self, data):
        for widget in self.marketing_business_figure_frame.winfo_children():
            widget.destroy()
        
        marketing_figure = Figure(figsize=(10, 4), dpi=100)
        marketing_ax3 = marketing_figure.add_subplot(121)
        marketing_ax4 = marketing_figure.add_subplot(122)

        if 'sales_data' in data:
            dates = [pd.to_datetime(d) for d in data['sales_data']['date']]
            values = data['sales_data']['value']
            marketing_ax3.bar(dates, values, color='g', label='Sales Data')
            marketing_ax3.set_title('Sales Data')

        if 'market_share' in data:
            labels = data['market_share']['labels']
            sizes = data['market_share']['sizes']
            marketing_ax4.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
            marketing_ax4.set_title('Market Share')

        marketing_ax3.set_xlabel('Date')
        marketing_ax3.set_ylabel('Value')
        marketing_ax3.legend()

        marketing_ax4.set_xlabel('Date')
        marketing_ax4.set_ylabel('Value')
        marketing_ax4.legend()

        marketing_canvas = FigureCanvasTkAgg(marketing_figure, self.marketing_business_figure_frame)
        marketing_canvas.draw()
        marketing_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def add_empty_plots(self, frame, title):
        figure = Figure(figsize=(10, 4), dpi=100)
        ax1 = figure.add_subplot(121)
        ax2 = figure.add_subplot(122)
        ax1.set_title(f'{title} 1')
        ax2.set_title(f'{title} 2')
        canvas = FigureCanvasTkAgg(figure, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

class HealthPage(SidebarFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.create_health_panel()

    def create_health_panel(self):
        health_frame1 = tk.Frame(self.scrollable_frame, pady=10, bg="white")
        health_frame1.pack(fill='x')
        health_label1 = tk.Label(health_frame1, text="Medical Data", font=('Helvetica', 16, 'bold'), bg="white")
        health_label1.pack(side='left')
        self.health_input_entry1 = tk.Entry(health_frame1)
        self.health_input_entry1.pack(side='left', padx=10)
        health_get_data_button1 = tk.Button(health_frame1, text="Get Data", command=self.fetch_medical_data, bg='white', fg='#4974a5')
        health_get_data_button1.pack(side='left', padx=10)
        health_load_dataset_button1 = tk.Button(health_frame1, text="Load Dataset", command=self.load_health_dataset, bg='white', fg='#4974a5')
        health_load_dataset_button1.pack(side='left', padx=10)
        
        self.health_medical_result_text = tk.Text(self.scrollable_frame, height=5, wrap='word', bg='#ECF1F6')
        self.health_medical_result_text.pack(fill='x', padx=10, pady=10)

        health_frame2 = tk.Frame(self.scrollable_frame, pady=10, bg="white")
        health_frame2.pack(fill='x')
        health_label2 = tk.Label(health_frame2, text="Healthsage Data", font=('Helvetica', 16, 'bold'), bg="white")
        health_label2.pack(side='left')
        self.health_input_entry2 = tk.Entry(health_frame2)
        self.health_input_entry2.pack(side='left', padx=10)
        health_get_data_button2 = tk.Button(health_frame2, text="Get Data", command=self.fetch_healthsage_data, bg='white', fg='#4974a5')
        health_get_data_button2.pack(side='left', padx=10)
        health_load_dataset_button2 = tk.Button(health_frame2, text="Load Dataset", command=self.load_health_dataset, bg='white', fg='#4974a5')
        health_load_dataset_button2.pack(side='left', padx=10)
        
        self.health_healthsage_result_text = tk.Text(self.scrollable_frame, height=5, wrap='word', bg='#ECF1F6')
        self.health_healthsage_result_text.pack(fill='x', padx=10, pady=10)

        trend_insights_frame = tk.LabelFrame(self.scrollable_frame, text="Trend Insights", bg="white", padx=10, pady=10)
        trend_insights_frame.pack(fill='both', expand=True, padx=10, pady=10)
        self.health_trend_figure_frame = tk.Frame(trend_insights_frame, bg="white")
        self.health_trend_figure_frame.pack(fill='both', expand=True, pady=10)

        business_data_insights_frame = tk.LabelFrame(self.scrollable_frame, text="Business Data Insights", bg="white", padx=10, pady=10)
        business_data_insights_frame.pack(fill='both', expand=True, padx=10, pady=10)
        self.health_business_figure_frame = tk.Frame(business_data_insights_frame, bg="white")
        self.health_business_figure_frame.pack(fill='both', expand=True, pady=10)

        # Add empty plots
        self.add_empty_plots(self.health_trend_figure_frame, "Health Trend Plot")
        self.add_empty_plots(self.health_trend_figure_frame, "Health Trend Plot")
        self.add_empty_plots(self.health_business_figure_frame, "Health Business Plot")
        self.add_empty_plots(self.health_business_figure_frame, "Health Business Plot")

    def fetch_medical_data(self):
        params = {"param": self.health_input_entry1.get()}
        try:
            data = med.get_medical_data(params)
            self.health_medical_result_text.delete(1.0, tk.END)
            self.health_medical_result_text.insert(tk.END, str(data))
            self.update_health_trend_plot(data)
            self.update_health_business_plot(data)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to fetch medical data: {e}")

    def fetch_healthsage_data(self):
        params = {"param": self.health_input_entry2.get()}
        try:
            data = healthsg.get_healthsage_data(params)
            self.health_healthsage_result_text.delete(1.0, tk.END)
            self.health_healthsage_result_text.insert(tk.END, str(data))
            self.update_health_trend_plot(data)
            self.update_health_business_plot(data)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to fetch healthsage data: {e}")

    def load_health_dataset(self):
        dialog = DataInputDialog(self)
        if dialog.variables and dialog.data_type and dialog.charts and dialog.models:
            data = {"variables": dialog.variables, "data_type": dialog.data_type, "charts": dialog.charts, "models": dialog.models}
            self.plot_health_custom_data(data)

    def plot_health_custom_data(self, data):
        pass

    def update_health_trend_plot(self, data):
        for widget in self.health_trend_figure_frame.winfo_children():
            widget.destroy()
        
        health_figure = Figure(figsize=(10, 4), dpi=100)
        health_ax1 = health_figure.add_subplot(121)
        health_ax2 = health_figure.add_subplot(122)
        
        if 'patient_data' in data:
            dates = [pd.to_datetime(d) for d in data['patient_data']['date']]
            values = data['patient_data']['value']
            health_ax1.plot(dates, values, marker='o', linestyle='-', color='b', label='Patient Data')
            health_ax1.set_title('Patient Data')
        
        if 'health_metrics' in data:
            dates = [pd.to_datetime(d) for d in data['health_metrics']['date']]
            values = data['health_metrics']['value']
            health_ax2.plot(dates, values, marker='x', linestyle='--', color='r', label='Health Metrics')
            health_ax2.set_title('Health Metrics')

        health_ax1.set_xlabel('Date')
        health_ax1.set_ylabel('Value')
        health_ax1.legend()

        health_ax2.set_xlabel('Date')
        health_ax2.set_ylabel('Value')
        health_ax2.legend()

        health_canvas = FigureCanvasTkAgg(health_figure, self.health_trend_figure_frame)
        health_canvas.draw()
        health_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def update_health_business_plot(self, data):
        for widget in self.health_business_figure_frame.winfo_children():
            widget.destroy()
        
        health_figure = Figure(figsize=(10, 4), dpi=100)
        health_ax3 = health_figure.add_subplot(121)
        health_ax4 = health_figure.add_subplot(122)

        if 'hospital_admissions' in data:
            dates = [pd.to_datetime(d) for d in data['hospital_admissions']['date']]
            values = data['hospital_admissions']['value']
            health_ax3.bar(dates, values, color='g', label='Hospital Admissions')
            health_ax3.set_title('Hospital Admissions')

        if 'disease_spread' in data:
            labels = data['disease_spread']['labels']
            sizes = data['disease_spread']['sizes']
            health_ax4.pie(sizes, labels=labels, autopct='%1.1%%', startangle=140)
            health_ax4.set_title('Disease Spread')

        health_ax3.set_xlabel('Date')
        health_ax3.set_ylabel('Value')
        health_ax3.legend()

        health_ax4.set_xlabel('Date')
        health_ax4.set_ylabel('Value')
        health_ax4.legend()

        health_canvas = FigureCanvasTkAgg(health_figure, self.health_business_figure_frame)
        health_canvas.draw()
        health_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def add_empty_plots(self, frame, title):
        figure = Figure(figsize=(10, 4), dpi=100)
        ax1 = figure.add_subplot(121)
        ax2 = figure.add_subplot(122)
        ax1.set_title(f'{title} 1')
        ax2.set_title(f'{title} 2')
        canvas = FigureCanvasTkAgg(figure, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

class SupplyChainPage(SidebarFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        self.create_supply_chain_panel()

    def create_supply_chain_panel(self):
        supply_chain_frame = tk.Frame(self.scrollable_frame, pady=10, bg="white")
        supply_chain_frame.pack(fill='x')
        supply_chain_label = tk.Label(supply_chain_frame, text="Supply Chain Data", font=('Helvetica', 16, 'bold'), bg="white")
        supply_chain_label.pack(side='left')
        self.supply_chain_input_entry = tk.Entry(supply_chain_frame)
        self.supply_chain_input_entry.pack(side='left', padx=10)
        supply_chain_get_data_button = tk.Button(supply_chain_frame, text="Get Data", command=self.fetch_supply_chain_data, bg='white', fg='#4974a5')
        supply_chain_get_data_button.pack(side='left', padx=10)
        supply_chain_load_dataset_button = tk.Button(supply_chain_frame, text="Load Dataset", command=self.load_supply_chain_dataset, bg='white', fg='#4974a5')
        supply_chain_load_dataset_button.pack(side='left', padx=10)
        
        self.supply_chain_result_text = tk.Text(self.scrollable_frame, height=10, wrap='word', bg='#ECF1F6')
        self.supply_chain_result_text.pack(fill='x', padx=10, pady=10)

        trend_insights_frame = tk.LabelFrame(self.scrollable_frame, text="Trend Insights", bg="white", padx=10, pady=10)
        trend_insights_frame.pack(fill='both', expand=True, padx=10, pady=10)
        self.supply_chain_trend_figure_frame = tk.Frame(trend_insights_frame, bg="white")
        self.supply_chain_trend_figure_frame.pack(fill='both', expand=True, pady=10)

        business_data_insights_frame = tk.LabelFrame(self.scrollable_frame, text="Business Data Insights", bg="white", padx=10, pady=10)
        business_data_insights_frame.pack(fill='both', expand=True, padx=10, pady=10)
        self.supply_chain_business_figure_frazme = tk.Frame(business_data_insights_frame, bg="white")
        self.supply_chain_business_figure_frame.pack(fill='both', expand=True, pady=10)

        # Add empty plots
        self.add_empty_plots(self.supply_chain_trend_figure_frame, "Supply Chain Trend Plot")
        self.add_empty_plots(self.supply_chain_trend_figure_frame, "Supply Chain Trend Plot")
        self.add_empty_plots(self.supply_chain_business_figure_frame, "Supply Chain Business Plot")
        self.add_empty_plots(self.supply_chain_business_figure_frame, "Supply Chain Business Plot")

    def fetch_supply_chain_data(self):
        params = {"param": self.supply_chain_input_entry.get()}
        try:
            data = supp_chain_opt.get_supply_chain_data(params)
            self.supply_chain_result_text.delete(1.0, tk.END)
            self.supply_chain_result_text.insert(tk.END, str(data))
            self.update_supply_chain_trend_plot(data)
            self.update_supply_chain_business_plot(data)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to fetch supply chain data: {e}")

    def load_supply_chain_dataset(self):
        dialog = DataInputDialog(self)
        if dialog.variables and dialog.data_type and dialog.charts and dialog.models:
            data = {"variables": dialog.variables, "data_type": dialog.data_type, "charts": dialog.charts, "models": dialog.models}
            self.plot_supply_chain_custom_data(data)

    def plot_supply_chain_custom_data(self, data):
        pass

    def update_supply_chain_trend_plot(self, data):
        for widget in self.supply_chain_trend_figure_frame.winfo_children():
            widget.destroy()
        
        supply_chain_figure = Figure(figsize=(10, 4), dpi=100)
        supply_chain_ax1 = supply_chain_figure.add_subplot(121)
        supply_chain_ax2 = supply_chain_figure.add_subplot(122)

        if 'inventory_levels' in data:
            dates = [pd.to_datetime(d) for d in data['inventory_levels']['date']]
            values = data['inventory_levels']['value']
            supply_chain_ax1.plot(dates, values, marker='o', linestyle='-', color='b', label='Inventory Levels')
            supply_chain_ax1.set_title('Inventory Levels')
        
        if 'delivery_times' in data:
            dates = [pd.to_datetime(d) for d in data['delivery_times']['date']]
            values = data['delivery_times']['value']
            supply_chain_ax2.plot(dates, values, marker='x', linestyle='--', color='r', label='Delivery Times')
            supply_chain_ax2.set_title('Delivery Times')

        supply_chain_ax1.set_xlabel('Date')
        supply_chain_ax1.set_ylabel('Value')
        supply_chain_ax1.legend()

        supply_chain_ax2.set_xlabel('Date')
        supply_chain_ax2.set_ylabel('Value')
        supply_chain_ax2.legend()

        supply_chain_canvas = FigureCanvasTkAgg(supply_chain_figure, self.supply_chain_trend_figure_frame)
        supply_chain_canvas.draw()
        supply_chain_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def update_supply_chain_business_plot(self, data):
        for widget in self.supply_chain_business_figure_frame.winfo_children():
            widget.destroy()
        
        supply_chain_figure = Figure(figsize=(10, 4), dpi=100)
        supply_chain_ax3 = supply_chain_figure.add_subplot(121)
        supply_chain_ax4 = supply_chain_figure.add_subplot(122)

        if 'demand_forecast' in data:
            dates = [pd.to_datetime(d) for d in data['demand_forecast']['date']]
            values = data['demand_forecast']['value']
            supply_chain_ax3.bar(dates, values, color='g', label='Demand Forecast')
            supply_chain_ax3.set_title('Demand Forecast')

        if 'supply_chain_costs' in data:
            labels = data['supply_chain_costs']['labels']
            sizes = data['supply_chain_costs']['sizes']
            supply_chain_ax4.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
            supply_chain_ax4.set_title('Supply Chain Costs')

        supply_chain_ax3.set_xlabel('Date')
        supply_chain_ax3.set_ylabel('Value')
        supply_chain_ax3.legend()

        supply_chain_ax4.set_xlabel('Date')
        supply_chain_ax4.set_ylabel('Value')
        supply_chain_ax4.legend()

        supply_chain_canvas = FigureCanvasTkAgg(supply_chain_figure, self.supply_chain_business_figure_frame)
        supply_chain_canvas.draw()
        supply_chain_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def add_empty_plots(self, frame, title):
        figure = Figure(figsize=(10, 4), dpi=100)
        ax1 = figure.add_subplot(121)
        ax2 = figure.add_subplot(122)
        ax1.set_title(f'{title} 1')
        ax2.set_title(f'{title} 2')
        canvas = FigureCanvasTkAgg(figure, frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Example mock functions for external dependencies to make the script self-contained and testable
    def get_mock_data(params):
        return {
            "stock_prices": {"date": ["2023-01-01", "2023-02-01"], "value": [100, 110]},
            "economic_indicators": {"date": ["2023-01-01", "2023-02-01"], "value": [200, 190]},
            "revenue": {"date": ["2023-01-01", "2023-02-01"], "value": [300, 310]},
            "profit_margins": {"labels": ["Product A", "Product B"], "sizes": [60, 40]},
            "campaign_performance": {"date": ["2023-01-01", "2023-02-01"], "value": [50, 55]},
            "social_media_metrics": {"date": ["2023-01-01", "2023-02-01"], "value": [150, 145]},
            "patient_data": {"date": ["2023-01-01", "2023-02-01"], "value": [70, 65]},
            "health_metrics": {"date": ["2023-01-01", "2023-02-01"], "value": [90, 85]},
            "inventory_levels": {"date": ["2023-01-01", "2023-02-01"], "value": [400, 390]},
            "delivery_times": {"date": ["2023-01-01", "2023-02-01"], "value": [30, 25]},
            "demand_forecast": {"date": ["2023-01-01", "2023-02-01"], "value": [500, 510]},
            "supply_chain_costs": {"labels": ["Logistics", "Production"], "sizes": [70, 30]}
        }

    # Mock implementations for fetching data
    fin = newscatcher = med = healthsg = supp_chain_opt = type('mock', (), {'get_financial_data': get_mock_data, 'get_news_data': get_mock_data, 'get_medical_data': get_mock_data, 'get_healthsage_data': get_mock_data, 'get_supply_chain_data': get_mock_data})()

# ---------------------------------------------------------------------------------------

class ReportsPage(SidebarFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        tk.Label(self.scrollable_frame, text="Reports", font=('Helvetica', 18), bg="white", fg="#4974a5").pack(pady=20)
        self.create_report_tools()

    def create_report_tools(self):
        tk.Button(self.scrollable_frame, text="Load Dataset", command=self.load_dataset).pack(pady=10)

        self.dataset_options = ["Dataset 1", "Dataset 2", "Dataset 3", "Loaded Dataset"]
        self.selected_dataset = tk.StringVar(self.scrollable_frame)
        self.selected_dataset.set(self.dataset_options[0])  # Set default option

        tk.Label(self.scrollable_frame, text="Select Dataset", bg="white").pack(pady=5)
        self.dataset_menu = tk.OptionMenu(self.scrollable_frame, self.selected_dataset, *self.dataset_options)
        self.dataset_menu.pack(pady=5)

        tk.Button(self.scrollable_frame, text="Generate Summary Report", command=self.generate_summary_report).pack(pady=10)
        tk.Button(self.scrollable_frame, text="Create Visual Report", command=self.create_visual_report).pack(pady=10)
        tk.Button(self.scrollable_frame, text="Export Report", command=self.export_report).pack(pady=10)
        tk.Button(self.scrollable_frame, text="Load Previous Report", command=self.load_previous_report).pack(pady=10)
        self.report_text = tk.Text(self.scrollable_frame, height=10, wrap='word')
        self.report_text.pack(fill='x', padx=5, pady=5)

    def load_dataset(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")])
        if file_path:
            try:
                if file_path.endswith('.csv'):
                    data = pd.read_csv(file_path)
                elif file_path.endswith('.xlsx'):
                    data = pd.read_excel(file_path)
                else:
                    messagebox.showerror("Invalid file", "Only CSV and Excel files are supported.")
                    return
                
                self.controller.data = data
                self.dataset_options[-1] = "Loaded Dataset"
                self.selected_dataset.set("Loaded Dataset")
                self.dataset_menu['menu'].entryconfig(3, label="Loaded Dataset")
                messagebox.showinfo("Info", "Dataset loaded successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load dataset: {e}")

    def generate_summary_report(self):
        dataset_name = self.selected_dataset.get()
        data = self.load_data(dataset_name)
        if data is not None:
            summary = data.describe()
            self.controller.report_data = summary
            self.report_text.delete(1.0, tk.END)
            self.report_text.insert(tk.END, summary.to_string())
            messagebox.showinfo("Info", "Summary report generated.")
        else:
            messagebox.showwarning("Warning", "No data available.")

    def create_visual_report(self):
        dataset_name = self.selected_dataset.get()
        data = self.load_data(dataset_name)
        if data is not None:
            figure = plt.Figure(figsize=(5, 4), dpi=100)
            ax = figure.add_subplot(111)
            data.hist(ax=ax)
            canvas = FigureCanvasTkAgg(figure, self.scrollable_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            messagebox.showinfo("Info", "Visual report created.")
        else:
            messagebox.showwarning("Warning", "No data available.")

    def export_report(self):
        if hasattr(self.controller, 'report_data'):
            file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if file_path:
                self.controller.report_data.to_csv(file_path)
                messagebox.showinfo("Info", "Report exported.")
        else:
            messagebox.showwarning("Warning", "No report data to export.")

    def load_previous_report(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            self.controller.report_data = pd.read_csv(file_path)
            self.report_text.delete(1.0, tk.END)
            self.report_text.insert(tk.END, self.controller.report_data.to_string())
            messagebox.showinfo("Info", "Previous report loaded.")

    def load_data(self, dataset_name):
        if dataset_name == "Dataset 1":
            data = pd.DataFrame(np.random.randn(100, 4), columns=list('ABCD'))
        elif dataset_name == "Dataset 2":
            data = pd.DataFrame(np.random.randn(200, 4), columns=list('WXYZ'))
        elif dataset_name == "Dataset 3":
            data = pd.DataFrame(np.random.randn(150, 4), columns=list('EFGH'))
        elif dataset_name == "Loaded Dataset" and hasattr(self.controller, 'data'):
            data = self.controller.data
        else:
            data = None
        return data

class DataAnalysisPage(SidebarFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        tk.Label(self.scrollable_frame, text="Data Analysis Tools", font=('Helvetica', 18), bg="white", fg="#4974a5").pack(pady=20)
        self.create_analysis_tools()

    def create_analysis_tools(self):
        tk.Button(self.scrollable_frame, text="Perform Statistical Analysis", command=self.perform_statistical_analysis).pack(pady=10)
        tk.Button(self.scrollable_frame, text="Show Correlation Matrix", command=self.show_correlation_matrix).pack(pady=10)
        tk.Button(self.scrollable_frame, text="Data Cleaning Tools", command=self.data_cleaning_tools).pack(pady=10)
        tk.Button(self.scrollable_frame, text="Perform PCA", command=self.perform_pca).pack(pady=10)
        self.analysis_text = tk.Text(self.scrollable_frame, height=10, wrap='word')
        self.analysis_text.pack(fill='x', padx=5, pady=5)

    def perform_statistical_analysis(self):
        if hasattr(self.controller, 'data'):
            data = self.controller.data
            stat_results = data.describe()
            self.analysis_text.delete(1.0, tk.END)
            self.analysis_text.insert(tk.END, stat_results.to_string())
            messagebox.showinfo("Info", "Statistical analysis performed.")
        else:
            messagebox.showwarning("Warning", "No data available.")

    def show_correlation_matrix(self):
        if hasattr(self.controller, 'data'):
            data = self.controller.data
            corr_matrix = data.corr()
            self.analysis_text.delete(1.0, tk.END)
            self.analysis_text.insert(tk.END, corr_matrix.to_string())
            messagebox.showinfo("Info", "Correlation matrix displayed.")
        else:
            messagebox.showwarning("Warning", "No data available.")

    def data_cleaning_tools(self):
        if hasattr(self.controller, 'data'):
            self.controller.data.dropna(inplace=True)
            self.controller.data.drop_duplicates(inplace=True)
            messagebox.showinfo("Info", "Data cleaned.")
        else:
            messagebox.showwarning("Warning", "No data available.")

    def perform_pca(self):
        if hasattr(self.controller, 'data'):
            data = self.controller.data
            numeric_data = data.select_dtypes(include=[np.number])
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(numeric_data)
            self.analysis_text.delete(1.0, tk.END)
            self.analysis_text.insert(tk.END, f"PCA Result: {pca_result}")
            messagebox.showinfo("Info", "PCA performed.")
        else:
            messagebox.showwarning("Warning", "No data available.")

class ModelSelectionPage(SidebarFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        tk.Label(self.scrollable_frame, text="Model Selection", font=('Helvetica', 18), bg="white", fg="#4974a5").pack(pady=20)
        self.create_model_selection_tools()

    def create_model_selection_tools(self):
        tk.Button(self.scrollable_frame, text="Compare Algorithms", command=self.compare_algorithms).pack(pady=10)
        tk.Button(self.scrollable_frame, text="Hyperparameter Tuning", command=self.hyperparameter_tuning).pack(pady=10)
        tk.Button(self.scrollable_frame, text="Perform Cross-Validation", command=self.perform_cross_validation).pack(pady=10)
        tk.Button(self.scrollable_frame, text="Show Model Summary", command=self.show_model_summary).pack(pady=10)
        self.model_selection_text = tk.Text(self.scrollable_frame, height=10, wrap='word')
        self.model_selection_text.pack(fill='x', padx=5, pady=5)

    def compare_algorithms(self):
        if hasattr(self.controller, 'data'):
            data = self.controller.data
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            models = {
                'DecisionTree': DecisionTreeClassifier(),
                # Add other models here
            }
            results = {}
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                results[name] = {
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred, average='weighted'),
                    'Recall': recall_score(y_test, y_pred, average='weighted'),
                    'F1 Score': f1_score(y_test, y_pred, average='weighted')
                }
            self.model_selection_text.delete(1.0, tk.END)
            self.model_selection_text.insert(tk.END, str(results))
            messagebox.showinfo("Info", "Algorithms compared.")
        else:
            messagebox.showwarning("Warning", "No data available.")

    def hyperparameter_tuning(self):
        if hasattr(self.controller, 'data'):
            data = self.controller.data
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            param_grid = {
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10]
            }
            grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=3, scoring='accuracy')
            grid_search.fit(X_train, y_train)
            best_params = grid_search.best_params_
            self.model_selection_text.delete(1.0, tk.END)
            self.model_selection_text.insert(tk.END, f"Best Params: {best_params}")
            messagebox.showinfo("Info", "Hyperparameter tuning performed.")
        else:
            messagebox.showwarning("Warning", "No data available.")

    def perform_cross_validation(self):
        if hasattr(self.controller, 'data'):
            data = self.controller.data
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
            model = DecisionTreeClassifier()
            scores = cross_val_score(model, X, y, cv=5)
            self.model_selection_text.delete(1.0, tk.END)
            self.model_selection_text.insert(tk.END, f"Cross-Validation Scores: {scores}")
            messagebox.showinfo("Info", "Cross-validation performed.")
        else:
            messagebox.showwarning("Warning", "No data available.")

    def show_model_summary(self):
        if hasattr(self.controller, 'model'):
            model = self.controller.model
            summary = str(model)
            self.model_selection_text.delete(1.0, tk.END)
            self.model_selection_text.insert(tk.END, summary)
            messagebox.showinfo("Info", "Model summary displayed.")
        else:
            messagebox.showwarning("Warning", "No model available.")

class ModelTrainingPage(SidebarFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        tk.Label(self.scrollable_frame, text="Model Training", font=('Helvetica', 18), bg="white", fg="#4974a5").pack(pady=20)
        self.create_training_tools()

    def create_training_tools(self):
        tk.Label(self.scrollable_frame, text="Select Model", bg="white").pack(pady=5)
        self.model_options = ["Decision Tree", "Random Forest", "SVM"]
        self.selected_model = tk.StringVar(self.scrollable_frame)
        self.selected_model.set(self.model_options[0])  # Set default option
        self.model_menu = tk.OptionMenu(self.scrollable_frame, self.selected_model, *self.model_options)
        self.model_menu.pack(pady=5)

        tk.Label(self.scrollable_frame, text="Set Parameters", bg="white").pack(pady=5)
        self.param_entry = tk.Entry(self.scrollable_frame)
        self.param_entry.pack(pady=5)
        self.param_entry.insert(0, "e.g. max_depth=3, n_estimators=100")

        tk.Button(self.scrollable_frame, text="Train Model", command=self.train_model).pack(pady=10)
        self.training_text = tk.Text(self.scrollable_frame, height=10, wrap='word')
        self.training_text.pack(fill='x', padx=5, pady=5)

    def train_model(self):
        dataset_name = "Loaded Dataset"  # Assuming we're using the loaded dataset
        data = self.controller.data if hasattr(self.controller, 'data') else None
        if data is not None:
            model_name = self.selected_model.get()
            params = self.param_entry.get()
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = None
            if model_name == "Decision Tree":
                model = DecisionTreeClassifier()
            elif model_name == "Random Forest":
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier()
            elif model_name == "SVM":
                from sklearn.svm import SVC
                model = SVC()

            if model:
                if params:
                    param_dict = {param.split('=')[0]: eval(param.split('=')[1]) for param in params.split(',')}
                    model.set_params(**param_dict)
                model.fit(X_train, y_train)
                self.controller.model = model
                self.training_text.delete(1.0, tk.END)
                self.training_text.insert(tk.END, f"{model_name} model trained successfully with parameters: {params}")
                messagebox.showinfo("Info", "Model trained successfully.")
        else:
            messagebox.showwarning("Warning", "No data available for training.")


class ModelEvaluationPage(SidebarFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        tk.Label(self.scrollable_frame, text="Model Evaluation", font=('Helvetica', 18), bg="white", fg="#4974a5").pack(pady=20)
        self.create_evaluation_tools()

    def create_evaluation_tools(self):
        tk.Button(self.scrollable_frame, text="Evaluate Model", command=self.evaluate_model).pack(pady=10)
        self.evaluation_text = tk.Text(self.scrollable_frame, height=10, wrap='word')
        self.evaluation_text.pack(fill='x', padx=5, pady=5)

    def evaluate_model(self):
        data = self.controller.data if hasattr(self.controller, 'data') else None
        model = self.controller.model if hasattr(self.controller, 'model') else None
        if data is not None and model is not None:
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            y_pred = model.predict(X_test)
            evaluation_results = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, average='weighted'),
                'Recall': recall_score(y_test, y_pred, average='weighted'),
                'F1 Score': f1_score(y_test, y_pred, average='weighted')
            }
            self.evaluation_text.delete(1.0, tk.END)
            self.evaluation_text.insert(tk.END, str(evaluation_results))
            messagebox.showinfo("Info", "Model evaluated successfully.")
        else:
            messagebox.showwarning("Warning", "No model available for evaluation.")


class ChatPage(SidebarFrame):
    def __init__(self, parent, controller):
        super().__init__(parent, controller)
        tk.Label(self.scrollable_frame, text="Chat", font=('Helvetica', 18), bg="white", fg="#4974a5").pack(pady=20)
        self.create_chat_interface()

    def create_chat_interface(self):
        self.chat_text = tk.Text(self.scrollable_frame, height=15, wrap='word')
        self.chat_text.pack(fill='x', padx=5, pady=5)
        self.chat_entry = tk.Entry(self.scrollable_frame)
        self.chat_entry.pack(fill='x', padx=5, pady=5)
        tk.Button(self.scrollable_frame, text="Send", command=self.send_message).pack(pady=10)
        tk.Button(self.scrollable_frame, text="Back", command=lambda: self.controller.show_frame("DashboardPage")).pack(pady=10)

    def send_message(self):
        message = self.chat_entry.get()
        if message:
            self.chat_text.insert(tk.END, f"You: {message}\n")
            self.chat_entry.delete(0, tk.END)
            # Placeholder for sending the message to a chatbot or other users
            self.chat_text.insert(tk.END, f"Bot: {self.generate_response(message)}\n")

    def generate_response(self, message):
        # Placeholder for generating a response from a chatbot
        return "This is a placeholder response."
    

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
            self.data = self.data[(self.data[col] >= mean - 3 * std) & (self.data[col] <= mean + 3 * std)]

        self.save_version()
        self.display_data(self.data)

    def transform_data(self):
        if self.data is None:
            messagebox.showwarning("No data", "Please upload data first.")
            return

        # Example transformation (standard scaling)
        numeric_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        scaler = StandardScaler()
        self.data[numeric_cols] = scaler.fit_transform(self.data[numeric_cols])

        self.save_version()
        self.display_data(self.data)

    def display_data(self, data):
        self.data_text.delete(1.0, tk.END)
        self.data_text.insert(tk.END, data.head().to_string())

    def revert_version(self):
        if self.current_version_index > 0:
            self.current_version_index -= 1
            self.data = self.data_versions[self.current_version_index]
            self.display_data(self.data)
            messagebox.showinfo("Version Reverted", "Reverted to previous data version.")
        else:
            messagebox.showwarning("No previous version", "No previous version to revert to.")

if __name__ == "__main__":
    app = AIDSSApp()
    app.mainloop()
    