from datetime import datetime

canvas_sections = {
    "Background": "Describe the customer's goals and pains.",
    "Solution": "Define the solution (features, integrations, constraints, and out-of-scope).",
    "Data": "Describe data sources (training, production) and the labeling process.",
    "Modeling": "Outline the iterative approach to modeling the task.",
    "Feedback": "Identify feedback sources for iteration.",
    "Value Proposition": "State the product's value and the pains it alleviates.",
    "Metrics": "List the key metrics to evaluate success.",
    "Inference": "Specify the type of inference (batch or real-time).",
    "Project": "Define the team, deliverables, and timelines.",
    "Feasibility": "Discuss project feasibility and required resources.",
    "Objectives": "List the key objectives for delivery.",
    "Evaluation": "Detail the offline and online evaluation methods."
}

def get_multiline_input(prompt):
    print(f"{prompt}\n(Type each point and press Enter. Press Enter twice to end.)")
    lines = []
    while True:
        line = input("> ")
        if line == "":
            break
        lines.append(f"- {line}")
    return "\n".join(lines) if lines else "Not specified"

def collect_canvas_info():
    canvas_data = {}
    print("Machine Learning Canvas - Fill out the information for each section.")
    
    try:
        import mlflow
        
        experiment_name = "Machine_Learning_Canvas_Experiment"
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run():
            for section, prompt in canvas_sections.items():
                print(f"\n{section}:\n{prompt}")
                response = get_multiline_input(f"Enter details for {section}")
                canvas_data[section] = response
                
                mlflow.log_param(section, response)
                
                if mlflow.get_param(section) is not None:
                    mlflow_response = mlflow.get_param(section)
                    print(f"\nLogged to MLflow for {section}:\n{mlflow_response}")
                
                additional_input = input(f"Any additional notes for {section}? (Press Enter to skip)\n> ")
                if additional_input:
                    canvas_data[section] += f"\n\nAdditional Notes:\n{additional_input}"
                    mlflow.log_param(f"{section}_additional", additional_input)

    except ImportError:
        print("MLflow not found. Continuing without logging to MLflow.")
        for section, prompt in canvas_sections.items():
            print(f"\n{section}:\n{prompt}")
            response = get_multiline_input(f"Enter details for {section}")
            canvas_data[section] = response
            
            additional_input = input(f"Any additional notes for {section}? (Press Enter to skip)\n> ")
            if additional_input:
                canvas_data[section] += f"\n\nAdditional Notes:\n{additional_input}"

    return canvas_data

def generate_markdown(canvas_data):
    markdown_content = "# Machine Learning Canvas\n\n"
    
    for section, content in canvas_data.items():
        markdown_content += f"## {section}\n{content}\n\n"
    
    filename = f"ML_Canvas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(filename, 'w') as file:
        file.write(markdown_content)
    
    print(f"\nCanvas successfully generated in the file '{filename}'.")

canvas_data = collect_canvas_info()
generate_markdown(canvas_data)
