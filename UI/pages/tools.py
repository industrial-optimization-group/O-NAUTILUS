import plotly.graph_objects as go
import numpy as np


def create_navigator_plot(
    request, objective_name, have_x_label=False, have_y_label=False, legend=False
):
    total_steps = request.content["total_steps"]

    fig = go.Figure()
    fig.update_layout(height=200, margin={"t": 0})
    if have_x_label:
        fig.update_xaxes(title_text="Steps")
    if have_y_label:
        fig.update_yaxes(title_text=objective_name)
    fig.update_xaxes(range=[-0.5, total_steps])
    # Colors
    optimistic_boundary = "hsva(37,68,100,1)"
    optimistic_fill = "hsva(37,68,100,0.6)"
    known_boundary = "hsva(210,79,57,1)"
    known_fill = "hsva(210,79,57,0.6)"
    preference_line = "black"
    # Optimistic lower bound
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            name="Optimistic Lower Bound",
            showlegend=False,
            mode="lines",
            line_color=optimistic_boundary,
        )
    )
    # Optimistic upper bound
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            fill="tonexty",
            name="Optimistic Reachable Range",
            mode="lines",
            line_color=optimistic_boundary,
            fillcolor=optimistic_fill,
            showlegend=legend,
        )
    )
    # lower bound
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            name="Lower Bound",
            showlegend=False,
            mode="lines",
            line_color=known_boundary,
        )
    )
    # upper bound
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            fill="tonexty",
            name="Known Reachable Range",
            mode="lines",
            line_color=known_boundary,
            fillcolor=known_fill,
            showlegend=legend,
        )
    )
    # preference
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            name="Aspiration Level",
            showlegend=legend,
            mode="lines",
            line_dash="solid",
            line_color=preference_line,
        )
    )
    # ideal point
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            name="Utopian point",
            mode="lines",
            line_color="green",
            showlegend=legend,
        )
    )
    # nadir point
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            name="Nadir Point",
            mode="lines",
            line_color="red",
            showlegend=legend,
        )
    )
    fig = extend_navigator_plot(request, objective_name, fig)
    return fig


def extend_navigator_plot(request, objective_name, fig):
    content = request.content
    bounds = content["data"]
    ideal_nadir = content["dimensions_data"]
    preference_point = content["preference"]
    steps_taken = content["steps_taken"]
    if np.isnan(preference_point.values[0][0]):
        preference_point = ideal_nadir.loc["ideal"].to_frame(name=0).transpose()

    for trace in range(7):
        fig["data"][trace]["x"] += (steps_taken,)
    fig["data"][0]["y"] += (bounds[objective_name]["optimistic_lower_bound"],)
    fig["data"][1]["y"] += (bounds[objective_name]["optimistic_upper_bound"],)
    fig["data"][2]["y"] += (bounds[objective_name]["lower_bound"],)
    fig["data"][3]["y"] += (bounds[objective_name]["upper_bound"],)

    fig["data"][4]["y"] += (preference_point[objective_name][0],)
    fig["data"][5]["y"] += (ideal_nadir[objective_name]["ideal"],)
    fig["data"][6]["y"] += (ideal_nadir[objective_name]["nadir"],)
    return fig


def createscatter2d(request):
    non_dom = request.content["data"].values
    achievable = non_dom[request.content["achievable_ids"]]
    non_dom_opt = request.content["optimistic_data"].values
    achievable_opt = non_dom_opt[request.content["achievable_ids_opt"]]
    ideal = request.content["dimensions_data"].loc["ideal"]
    nadir = request.content["dimensions_data"].loc["nadir"]
    past_points = request.content["current_points_list"].values
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
    figure.add_trace(
        go.Scatter(
            x=past_points[:, 0],
            y=past_points[:, 1],
            name="Past Points",
            line_color="grey",
        )
    )
    figure.add_trace(go.Scatter(x=[current[0]], y=[current[1]], name="Current point"))
    figure.add_trace(
        go.Scatter(x=[preference[0]], y=[preference[1]], name="Preference")
    )
    return figure


def parallel_coordinates(data, dimensions, color):
    pass
