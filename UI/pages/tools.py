import plotly.graph_objects as go
import numpy as np

from plotly.subplots import make_subplots


def create_navigator_plot(request):
    ideal_nadir = request.content["dimensions_data"]
    total_steps = request.content["total_steps"]

    fig = make_subplots(
        rows=ideal_nadir.shape[1],
        cols=1,
        shared_xaxes=True,
        subplot_titles=list(ideal_nadir.columns),
    )
    fig.update_xaxes(title_text="Steps", row=2, col=1)
    fig.update_xaxes(range=[0, total_steps])
    for i in range(ideal_nadir.shape[1]):  # Looping over subplots/rows/objectives
        legend = True if i == 0 else False
        # Optimistic lower bound
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                name=f"Optimistic Lower Bound",
                showlegend=False,
                mode="lines+markers",
                line_color="yellow",
            ),
            row=i + 1,
            col=1,
        )
        # Optimistic upper bound
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                fill="tonexty",
                name=f"Optimistic Reachable area",
                mode="lines+markers",
                line_color="green",
                fillcolor="yellow",
                showlegend=legend,
            ),
            row=i + 1,
            col=1,
        )
        # lower bound
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                name=f"Lower Bound",
                showlegend=False,
                mode="lines+markers",
                line_color="green",
            ),
            row=i + 1,
            col=1,
        )
        # upper bound
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                fill="tonexty",
                name=f"Reachable area",
                mode="lines+markers",
                line_color="green",
                fillcolor="green",
                showlegend=legend,
            ),
            row=i + 1,
            col=1,
        )
        # preference
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                name=f"Preference",
                showlegend=legend,
                mode="lines+markers",
                line_dash="dash",
                line_color="brown",
            ),
            row=i + 1,
            col=1,
        )
        # ideal point
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                name=f"ideal",
                mode="lines+markers",
                line_dash="dash",
                line_color="green",
                showlegend=legend,
            ),
            row=i + 1,
            col=1,
        )
        # nadir point
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                name=f"nadir",
                mode="lines+markers",
                line_dash="dash",
                line_color="red",
                showlegend=legend,
            ),
            row=i + 1,
            col=1,
        )
    fig = extend_navigator_plot(request, fig)
    return fig


def extend_navigator_plot(request, fig):
    content = request.content
    bounds = content["data"]
    ideal_nadir = content["dimensions_data"]
    preference_point = content["preference"]
    steps_taken = content["steps_taken"]
    if np.isnan(preference_point.values[0][0]):
        preference_point = ideal_nadir.loc["nadir"].to_frame(name=0).transpose()

    for row, objective_name in enumerate(ideal_nadir.columns):
        for trace in range(7):
            fig["data"][7 * row + trace]["x"] += (steps_taken,)
        fig["data"][7 * row + 0]["y"] += (
            bounds[objective_name]["optimistic_lower_bound"],
        )
        fig["data"][7 * row + 1]["y"] += (
            bounds[objective_name]["optimistic_upper_bound"],
        )
        fig["data"][7 * row + 2]["y"] += (bounds[objective_name]["lower_bound"],)
        fig["data"][7 * row + 3]["y"] += (bounds[objective_name]["upper_bound"],)

        fig["data"][7 * row + 4]["y"] += (preference_point[objective_name][0],)
        fig["data"][7 * row + 5]["y"] += (ideal_nadir[objective_name]["ideal"],)
        fig["data"][7 * row + 6]["y"] += (ideal_nadir[objective_name]["nadir"],)
    return fig


def createscatter2d(request):
    non_dom = request.content["data"].values
    achievable = non_dom[request.content["achievable_ids"]]
    non_dom_opt = request.content["optimistic_data"].values
    achievable_opt = non_dom_opt[request.content["achievable_ids_opt"]]
    ideal = request.content["dimensions_data"].loc["ideal"]
    nadir = request.content["dimensions_data"].loc["nadir"]
    current = request.content["current_point"].values[0]
    preference = request.content["preference"].values[0]
    non_dom = non_dom[non_dom[:, 1].argsort()]
    achievable = achievable[achievable[:, 1].argsort()]
    non_dom_opt = non_dom_opt[non_dom_opt[:, 1].argsort()]
    achievable_opt = achievable_opt[achievable_opt[:, 1].argsort()]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=non_dom[:, 0],
            y=non_dom[:, 1],
            name="Non-dominated front",
            line_color="grey",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=achievable[:, 0],
            y=achievable[:, 1],
            name="Achievable front",
            line_color="green",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=non_dom_opt[:, 0],
            y=non_dom_opt[:, 1],
            name="Non-dominated optimistic front",
            line_color="black",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=achievable_opt[:, 0],
            y=achievable_opt[:, 1],
            name="Achievable optimistic front",
            line_color="yellow",
        )
    )
    figure.add_trace(go.Scatter(x=[ideal[0]], y=[ideal[1]], name="Ideal point"))
    figure.add_trace(go.Scatter(x=[nadir[0]], y=[nadir[1]], name="Nadir point"))
    figure.add_trace(go.Scatter(x=[current[0]], y=[current[1]], name="Current Point"))
    figure.add_trace(
        go.Scatter(x=[preference[0]], y=[preference[1]], name="Preference")
    )
    return figure
