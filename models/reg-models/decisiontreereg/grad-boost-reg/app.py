from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def home():
    # Test data
    
    data = [
        ("01-01-2020", 1597),
        ("02-01-2020", 1456),
        ("03-01-2020", 1908),
        ("04-01-2020", 896),
        ("05-01-2020", 755),
        ("06-01-2020", 453),
        ("07-01-2020", 1100),
        ("08-01-2020", 1235),
        ("09-01-2020", 1478),
    ]
    labels = [row[0] for row in data]
    values = [row[1] for row in data]
    return render_template("logist_reg.html", labels=labels, values=values)
