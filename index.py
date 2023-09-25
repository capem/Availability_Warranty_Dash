from dash import dcc, html
from dash.dependencies import Input, Output
from waitress import serve
from app import app
from apps import adjust, app1, index_page, results
import logging

logging.basicConfig(level=logging.INFO)

app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        html.Div(id="page-content", style={"height": "100vh"}),
    ]
)


# Use a dictionary to map paths to layouts
path_layouts = {
    "/": index_page.layout,
    "/apps/app1": app1.layout,
    "/apps/results": results.layout,
    "/apps/adjust": adjust.layout
}

@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def display_page(pathname):
    return path_layouts.get(pathname, "404")


if __name__ == "__main__":
    
    app.run_server(debug=True, port=80, host="localhost")
    # app.run_server()

    # serve(app.server, host="localhost", port=80, threads=6, channel_timeout=400)
    # waitress-serve --port=80 --host=localhost --channel-timeout=400 --threads=6 index:app.server
