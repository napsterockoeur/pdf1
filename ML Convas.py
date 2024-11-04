from datetime import datetime

# Order and structure of sections in the Machine Learning Canvas with prompts
canvas_sections = {
    "Background": "Describe the customer's goals and pains. (Press Enter twice to end the section)",
    "Solution": "Define the solution, including features, integrations, constraints, and out-of-scope items.",
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


# Capture inputs
def get_multiline_input(prompt):
    print(f"{prompt}\n(Type each point and press Enter. Press Enter twice to end.)")
    lines = []
    while True:
        line = input("> ")
        if line == "":
            break
        lines.append(f"- {line}")  # Format each line as a bullet point
    return "\n".join(lines) if lines else "Not specified"


# Interactive collection,
def collect_canvas_info():
    canvas_data = {}
    print("Machine Learning Canvas - Fill out the information for each section.")

    for section, prompt in canvas_sections.items():
        print(f"\n{section}:\n{prompt}")
        response = get_multiline_input(f"Enter details for {section}")
        canvas_data[section] = response

    return canvas_data


# Generate MD
def generate_markdown(canvas_data):
    markdown_content = "# Machine Learning Canvas\n\n"

    for section, content in canvas_data.items():
        markdown_content += f"## {section}\n{content}\n\n"

    filename = f"ML_Canvas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(filename, 'w') as file:
        file.write(markdown_content)

    print(f"\nCanvas successfully generated in the file '{filename}'.")


# Run
canvas_data = collect_canvas_info()
generate_markdown(canvas_data)
d