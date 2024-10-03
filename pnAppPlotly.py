import pandas as pd
import plotly.express as px
import panel as pn

pn.extension('plotly', 'tabulator')

def load_iris_data():
    return px.data.iris()

def load_election_data():
    return px.data.election()

def create_iris_section():
    df = load_iris_data()

    checkbox = pn.widgets.Checkbox(name='Show raw data (Iris)')
    columns = df.columns.tolist()
    x_axis = pn.widgets.Select(name='Select X axis', options=columns, value=columns[0])
    y_axis = pn.widgets.Select(name='Select Y axis', options=columns, value=columns[1])

    # Create the plot and table
    plot_pane = pn.pane.Plotly(sizing_mode='stretch_width', height=400)
    data_pane = pn.widgets.Tabulator(
        df,
        height=300,
        sizing_mode='stretch_width',
        theme='simple',
        layout='fit_columns'
    )
    data_pane.visible = False

    @pn.depends(x_axis.param.value, y_axis.param.value, watch=True)
    def update_iris_plot(x, y):
        fig = px.scatter(df, x=x, y=y, color="species")
        plot_pane.object = fig

    @pn.depends(checkbox.param.value, watch=True)
    def update_data_pane(show_data):
        data_pane.visible = show_data

    update_iris_plot(x_axis.value, y_axis.value)

    iris_section = pn.Column(
        pn.pane.Markdown("## Iris Dataset Analysis"),
        pn.Row(
            pn.Column(checkbox, sizing_mode='stretch_width'),
            pn.Column(x_axis, y_axis, sizing_mode='stretch_width'),
            sizing_mode='stretch_width'
        ),
        plot_pane,
        data_pane,
        sizing_mode='stretch_width',
        visible=False
    )

    return iris_section

def create_election_section():
    df = load_election_data()

    checkbox = pn.widgets.Checkbox(name='Show raw data (Election)')

    plot_pane = pn.pane.Plotly(sizing_mode='stretch_width', height=600)
    data_pane = pn.widgets.Tabulator(
        df,
        height=300,
        sizing_mode='stretch_width',
        theme='simple',
        layout='fit_data_fill'
    )
    data_pane.visible = False

    @pn.depends(checkbox.param.value, watch=True)
    def update_data_pane(show_data):
        data_pane.visible = show_data

    fig = px.scatter_3d(
        df, x="Joly", y="Coderre", z="Bergeron",
        color="winner", size="total", hover_name="district",
        symbol="result", color_discrete_map={
            "Joly": "blue", "Bergeron": "green", "Coderre": "red"
        }
    )
    plot_pane.object = fig

    election_section = pn.Column(
        pn.pane.Markdown("## Election Dataset Analysis"),
        checkbox,
        plot_pane,
        data_pane,
        sizing_mode='stretch_width',
        visible=False  # Initially hidden
    )

    return election_section

iris_section = create_iris_section()
election_section = create_election_section()

section_selector = pn.widgets.RadioButtonGroup(
    name='Select Analysis Section',
    options=['Iris Analysis', 'Election Analysis'],
    button_type='success'
)

def update_section(event):
    if event.new == 'Iris Analysis':
        iris_section.visible = True
        election_section.visible = False
    elif event.new == 'Election Analysis':
        iris_section.visible = False
        election_section.visible = True

iris_section.visible = True
election_section.visible = False

section_selector.param.watch(update_section, 'value')

template = pn.template.MaterialTemplate(title='Data Analysis Application')

template.sidebar.append(pn.pane.Markdown("# Menu"))
template.sidebar.append(section_selector)
template.main.append(iris_section)
template.main.append(election_section)

template.servable()
