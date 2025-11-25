import tkinter as tk
import subprocess

def run_shell_script():
    try:
        # Example: Running a simple 'ls' command
        result = subprocess.run(["ls", "-l"], capture_output=True, text=True, check=True)
        output_label.config(text=result.stdout)
    except subprocess.CalledProcessError as e:
        output_label.config(text=f"Error: {e.stderr}")

app = tk.Tk()
app.title("Shell Script Launcher")

run_button = tk.Button(app, text="Run Script", command=run_shell_script)
run_button.pack(pady=10)

output_label = tk.Label(app, text="")
output_label.pack(pady=5)

app.mainloop()