"""Dash entry point for the combined MAVEN/IUVS aurora explorer.

The module loads the detection catalog and magnetic-topology raster at import
time, builds the Explorer and MLR layouts, and connects them to Dash callbacks.
Local science-data paths are documented in ``README.md``.
"""

import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objects as go

from Spectrum.misc_helpers import *
from Spectrum.detector import *
from Spectrum.mlr_fit import *
from Spectrum.alt_profiles import get_alt_profiles

from Explorer.map_plotly import plot_map
from Explorer.sphere_map import plot_sphere
from Explorer.sza_plotly import sza_hist
from Explorer.lat_plotly import latitude_hist
from Explorer.coadded_plotly import coadd
from Explorer.coverage_plotly import plot_coverage
from Explorer.lt_plotly import plot_lt
from Explorer.nightside_plotly import plot_nightside
from Explorer.mso_coords import *


#Build check boxes for components to display
COMPONENT_OPTIONS = [
    {"label": "Measured spectrum", "value": "measured"},
    #{"label": "Background spectrum", "value": "background"},
    {"label": "Total MLR fit", "value": "total_fit"},
    {"label": "Solar continuum", "value": "solar"},
    {"label": "CO Cameron", "value": "co"},
    {"label": "CO₂⁺ UVD", "value": "co2p"},
    {"label": "NO nightglow", "value": "no"},
    {"label": "Constant", "value": "constant"},
]

#Add labels and line styles for each component
TRACE_NAMES = {
    "measured": "Measured spectrum",
    #"background": "Background spectrum",
    "total_fit": "Total MLR fit",
    "solar": "Solar continuum",
    "co": "CO Cameron",
    "co2p": "CO₂⁺ UVD",
    "no": "NO nightglow",
    "constant": "Constant",
}

LINE_STYLES = {
    "measured": {"width": 3},
    #"background": {"width": 3},
    "total_fit": {"width": 3, "dash": "dash"},
    "solar": {"width": 2},
    "co": {"width": 2},
    "co2p": {"width": 2},
    "no": {"width": 2},
    "constant": {"width": 2, "dash": "dot"},
}

# Scientific threshold used to separate anomalous and normal CO/CO2+ ratios.
coUVD = 12.7

# Expected spectral width. IUVS products encountered here use 512 or 219 bins.
wl_dim = 512

#master_path = r'D:/nightglow_tables/master.csv' #Huge csv with all data merged
# External catalog and repository-owned topology raster. Change ``master_path``
# to match the local data installation; see README.md for the expected schema.
master_path = r'E:/nightglow/master_cnn.csv'
map_img_dir = r'assets/bfield_smoothed.npy'
map_img = np.load(map_img_dir, allow_pickle=False)
data = pd.read_csv(master_path)
data = data.drop_duplicates(subset=["nick_number"])

data = data[
        (data['color_flag_muv'] == 'green') &
        (data['type']!='NON-DET') &
        (data['type'] != 'NONE') &
        (data['type'] != 'SEP')
    ]

lt = data['local_time'] % 24
data['hours_from_midnight'] = ((lt + 12) % 24) - 12
data=mso_coords(data)
data = data.drop_duplicates(subset=["nick_number"]) #Just to be safe

header_layout = html.Div([

        #Left side
        html.Div([
            html.H1("MARS AURORA EXPLORER",
                    style={
                        "margin": "0",
                        "fontSize": "26px",
                        "letterSpacing": "3px",
                    }),

            html.Div(
                "MAVEN Imaging Ultraviolet Spectrograph",
                style={
                    "color": "#7f8fa6",
                    "fontSize": "13px",
                    "letterSpacing": "1px",
                    "marginTop": "4px",
                }
            ),
        ]),

        #Right side
        html.Img(
            src="/assets/maven-logo-full.png",
            style={
            "height": "60px",
            "width": "180px",
            }
        ),
    ],
    style={
    "display": "flex",
    "justifyContent": "space-between",
    "alignItems": "center",
    "marginBottom": "18px",
    })

mlr_layout = html.Div(
    [
        #Header for the title page
        html.H2("IUVS MLR Spectrum Viewer"),

        #Section for all input fields
        html.Div([
                #Orbit input
                html.Div(
                    [
                        html.Label("Orbit"),
                        dcc.Input(
                            id="orbit-input",
                            type="number",
                            value=8675,
                            min=1,
                            step=1,
                            style={
                                "width": "80px",
                                "backgroundColor": "#080c16",
                                "color": "#e8edf5",
                                "border": "1px solid #34445c",
                                "borderRadius": "5px",
                                "padding": "7px",
                            },
                        ),
                    ]
                ),
                #Scan input
                html.Div(
                    [
                        html.Label("Scan"),
                        dcc.Input(
                            id="scan-input",
                            type="number",
                            value=3,
                            min=1,
                            step=1,
                            style={
                                "width": "80px",
                                "backgroundColor": "#080c16",
                                "color": "#e8edf5",
                                "border": "1px solid #34445c",
                                "borderRadius": "5px",
                                "padding": "7px",
                            },
                        ),
                    ]
                ),
                #Integration input
                html.Div([
                        html.Label("Integration"),
                        dcc.Input(
                            id="integration-input",
                            type="number",
                            value=10,
                            min=1,
                            step=1,
                            style={
                                "width": "80px",
                                "backgroundColor": "#080c16",
                                "color": "#e8edf5",
                                "border": "1px solid #34445c",
                                "borderRadius": "5px",
                                "padding": "7px",
                            },
                        ),
                    ]
                ),
                #Reload data button
                html.Button(
                    "Load spectrum",
                    id="load-button",
                    n_clicks=0,
                    style={
                    "backgroundColor": "#168aad",
                    "color": "#ffffff",
                    "border": "1px solid #2aa7c9",
                    "borderRadius": "5px",
                    "padding": "8px 18px",
                    "fontSize": "12px",
                    "fontWeight": "bold",
                    "letterSpacing": "1px",
                    "cursor": "pointer",
                    "height": "36px",
                    }
                ),
            ],
            style={
                "display": "flex",
                "alignItems": "end",
                "gap": "18px",
                "padding": "16px 20px",
                "backgroundColor": "#101827",
                "border": "1px solid #263348",
                "borderRadius": "10px",
                "marginBottom": "15px",
                },
        ),

        html.Label("Displayed components"),

        dcc.Checklist(
            id="component-checklist",
            options=COMPONENT_OPTIONS,
            value=[
                "measured",
                #"background",
                "total_fit",
                "solar",
                "co",
                "co2p",
                "no",
                "constant",
            ],
            inline=True,
            style={"marginBottom": "15px"},
            labelStyle={
                "color": "#c7d2e0",
                "marginRight": "18px",
                "fontSize": "13px",
                "letterSpacing": "0.3px",
            }
        ),

        html.Div(
            id="spectrum-information",
            style={"marginBottom": "10px"},
        ),
        #Error message display (if we have any)
        html.Div(
            id="error-message",
            style={
                "color": "red",
                "fontWeight": "bold",
                "marginBottom": "10px",
            },
        ),
        #Add graph section
        dcc.Graph(
            id="detector_image",
            config={
                "displaylogo": False,
                "scrollZoom": False,
            },
        ),

        dcc.Graph(
            id="spectrum-graph",
            config={
                "displaylogo": False,
                "scrollZoom": True,
            },
        ),

        #Two altitude profiles
        html.Div(
            [
                # Cameron bands
                html.Div(
                    dcc.Graph(
                        id="cam-alt-graph",
                        config={
                            "displaylogo": False,
                            "scrollZoom": True,
                            "responsive": True,
                        },
                        style={"height": "100%", "width": "100%"},
                    ),
                    id="cam_alt",
                    style={
                        "height": "60vh",
                        "width": "50%",
                        "minHeight": "350px",
                    },
                ),

                # UVD
                html.Div(
                    dcc.Graph(
                        id="uvd-alt-graph",
                        config={
                            "displaylogo": False,
                            "scrollZoom": True,
                            "responsive": True,
                        },
                        style={"height": "100%", "width": "100%"},
                    ),
                    id="uvd_alt",
                    style={
                        "height": "60vh",
                        "width": "50%",
                        "minHeight": "350px",
                    },
                ),
            ],
            style={
                "display": "flex",
                "width": "100%",
                "gap": "18px",
            },
        ),

        
    ],

    style={
        "maxWidth": "1400px",
        "margin": "30px auto",
        "padding": "0 25px",
    },
)

explorer_layout = html.Div([

    #Input fields section
    html.Div([

        #Orbit start
        html.Div([
            html.Label(
                "ORBIT START",
                style={
                    "fontSize": "11px",
                    "letterSpacing": "1px",
                    "color": "#8fa1b8",
                    "display": "block",
                    "marginBottom": "5px",
                }
            ),

            dcc.Input(
                id="orb_start_input",
                type="number",
                value=0,
                min=0,
                step=1,
                style={
                    "width": "80px",
                    "backgroundColor": "#080c16",
                    "color": "#e8edf5",
                    "border": "1px solid #34445c",
                    "borderRadius": "5px",
                    "padding": "7px",
                }
            ),
        ]),


        #Orbit end
        html.Div([
            html.Label(
                    "ORBIT END",
                    style={
                        "fontSize": "11px",
                        "letterSpacing": "1px",
                        "color": "#8fa1b8",
                        "display": "block",
                        "marginBottom": "5px",
                    }
                ),
        
                dcc.Input(
                    id="orb_end_input",
                    type="number",
                    value=24600,
                    min=0,
                    step=1,
                    style={
                        "width": "80px",
                        "backgroundColor": "#080c16",
                        "color": "#e8edf5",
                        "border": "1px solid #34445c",
                        "borderRadius": "5px",
                        "padding": "7px",
                    }
                ),
        ]),

        #Pearson R
        html.Div([
            html.Label(
                    "PEARSON R",
                    style={
                        "fontSize": "11px",
                        "letterSpacing": "1px",
                        "color": "#8fa1b8",
                        "display": "block",
                        "marginBottom": "5px",
                    }
                ),
        
                dcc.Input(
                    id="pearson",
                    type="number",
                    value=0.60,
                    min=0,
                    step=0.01,
                    style={
                        "width": "80px",
                        "backgroundColor": "#080c16",
                        "color": "#e8edf5",
                        "border": "1px solid #34445c",
                        "borderRadius": "5px",
                        "padding": "7px",
                    }
                ),
        ]),

        #Num of pixels
        html.Div([
            html.Label(
                    "MIN PIXELS",
                    style={
                        "fontSize": "11px",
                        "letterSpacing": "1px",
                        "color": "#8fa1b8",
                        "display": "block",
                        "marginBottom": "5px",
                    }
                ),
        
                dcc.Input(
                    id="n_pix",
                    type="number",
                    value=6,
                    min=1,
                    step=1,
                    style={
                        "width": "80px",
                        "backgroundColor": "#080c16",
                        "color": "#e8edf5",
                        "border": "1px solid #34445c",
                        "borderRadius": "5px",
                        "padding": "7px",
                    }
                ),
        ]),

        #SZA cutoff
        html.Div([
            html.Label(
                    "SZA MIN",
                    style={
                        "fontSize": "11px",
                        "letterSpacing": "1px",
                        "color": "#8fa1b8",
                        "display": "block",
                        "marginBottom": "5px",
                    }
                ),
        
                dcc.Input(
                    id="sza_min",
                    type="number",
                    value=105,
                    min=1,
                    step=1,
                    style={
                        "width": "80px",
                        "backgroundColor": "#080c16",
                        "color": "#e8edf5",
                        "border": "1px solid #34445c",
                        "borderRadius": "5px",
                        "padding": "7px",
                    }
                ),
        ]),


        #Detection type selector
        html.Div([
            html.Label(
                    "DETECTION TYPE",
                    style={
                        "fontSize": "11px",
                        "letterSpacing": "1px",
                        "color": "#8fa1b8",
                        "display": "block",
                        "marginBottom": "5px",
                    }
                ),

            dcc.Dropdown(
                id="type-selector",
                options=[
                    {"label": "Normal", "value": "norm"},
                    {"label": "Anomalous", "value": "anom"},
                    {"label": "All", "value": "all"},
                ],
                value="all", #Default selection
                clearable=False,
                className="theme-dropdown",
                style={
                    "width": "140px",
                    "fontSize": "13px",
                }
            ),
        ]),

        #Refresh button
        html.Div([
            html.Button(
                "Refresh",
                id="load_button",
                n_clicks=0,
                style={
                "backgroundColor": "#168aad",
                "color": "#ffffff",
                "border": "1px solid #2aa7c9",
                "borderRadius": "5px",
                "padding": "8px 18px",
                "fontSize": "12px",
                "fontWeight": "bold",
                "letterSpacing": "1px",
                "cursor": "pointer",
                "height": "36px",
                }
            )
        ]),
    ],
    style={
        "display": "flex",
        "alignItems": "end",
        "gap": "18px",
        "padding": "16px 20px",
        "backgroundColor": "#101827",
        "border": "1px solid #263348",
        "borderRadius": "10px",
        "marginBottom": "15px",
        }),

    #Plotting
    html.Div([

        #Map plot
        html.Div([
            dcc.Graph(
                id="map-image",
                config={
                    "displaylogo": False,
                    "scrollZoom": False,
                    "responsive": True,
                },
                style={"height": "100%",
                       "width": "100%",
                       },
            ),
        ],
        style={
        "height": "90vh",
        "minHeight": "450px",
        }
        ),

        #Sphere map plot
        html.Div([
            dcc.Graph(
                id="globe-image",
                config={
                    "displaylogo": False,
                    "scrollZoom": False,
                    "responsive": True,
                },
                style={"height": "100%",
                        "width": "100%",
                        },
            ),
        ],
        style={
        "height": "90vh",
        "minHeight": "450px",
        }
        ),


        #Lat + SZA plots
        html.Div([
            #SZA plot
            html.Div([
                dcc.Graph(
                    id="sza-image",
                    config={
                        "displaylogo": False,
                        "scrollZoom": False,
                        "responsive": True,
                    },
                    style={"height": "100%",
                            "width": "100%",
                            },
                ),
            ],
            style={
            "height": "60vh",
            "width": "50%",
            "minHeight": "350px",
            }
            ),

            #Latitude plot
            html.Div([
                dcc.Graph(
                    id="lat-image",
                    config={
                        "displaylogo": False,
                        "scrollZoom": False,
                        "responsive": True,
                    },
                    style={"height": "100%",
                            "width": "100%",
                            },
                ),
            ],
            style={
            "height": "60vh",
            "width": "50%",
            "minHeight": "350px",
            }
            ),
        ],
        style={
            "display": "flex",
            "width": "100%",
            "gap": "18px",
        }),

        #Spectra + LS plots
        html.Div([
            #Coadded spectra plot
            html.Div([
                dcc.Graph(
                    id="spectra-image",
                    config={
                        "displaylogo": False,
                        "scrollZoom": True,
                        "responsive": True,
                    },
                    style={"height": "100%",
                            "width": "100%",
                            },
                ),
            ],
            style={
            "height": "60vh",
            "width": "50%",
            "minHeight": "350px",
            }
            ),

            #LS vs year plot
            html.Div([
                dcc.Graph(
                    id="coverage-image",
                    config={
                        "displaylogo": False,
                        "scrollZoom": True,
                        "responsive": True,
                    },
                    style={"height": "100%",
                            "width": "100%",
                            },
                ),
            ],
            style={
            "height": "60vh",
            "width": "50%",
            "minHeight": "350px",
            }
            ),
        ],
        style={
            "display": "flex",
            "width": "100%",
            "gap": "18px",
        }),

        #Ls/ratio and LT hist
        html.Div([
            #LS vs ratio plot
            html.Div([
                dcc.Graph(
                    id="ls_ratio-image",
                    config={
                        "displaylogo": False,
                        "scrollZoom": True,
                        "responsive": True,
                    },
                    style={"height": "100%",
                            "width": "100%",
                            },
                ),
            ],
            style={
            "height": "60vh",
            "width": "50%",
            "minHeight": "350px",
            }
            ),

            #Local time histogram
            html.Div([
                dcc.Graph(
                    id="lt-image",
                    config={
                        "displaylogo": False,
                        "scrollZoom": True,
                        "responsive": True,
                    },
                    style={"height": "100%", 
                            "width": "100%",
                            },
                ),
            ],
            style={
            "height": "60vh",
            "width": "50%",
            "minHeight": "350px",
            }
            ),
        ],
        style={
            "display": "flex",
            "width": "100%",
            "gap": "18px",
        }
        ),

        #The two nightside plots
        html.Div([

            #Nightside polar plot
            html.Div([
                dcc.Graph(
                    id="nightside-polar",
                    config={
                        "displaylogo": False,
                        "scrollZoom": True,
                        "responsive": True,
                    },
                    style={
                        "height": "100%",
                        "width": "100%",
                    },
                ),
            ],
            style={
                "height": "60vh",
                "width": "50%",
                "minHeight": "350px",
            }),

            #Nightside cartesian plot
            html.Div([
                dcc.Graph(
                    id="nightside-cartesian",
                    config={
                        "displaylogo": False,
                        "scrollZoom": True,
                        "responsive": True,
                    },
                    style={
                        "height": "100%",
                        "width": "100%",
                    },
                ),
            ],
            style={
                "height": "60vh",
                "width": "50%",
                "minHeight": "350px",
            }),

        ],
        style={
            "display": "flex",
            "width": "100%",
            "gap": "18px",
        }),

        #MSO Cartesian plot + something later
        html.Div([
            #MSO Cart
            html.Div([
                dcc.Graph(
                    id="nightside-cartesian-mso",
                    config={
                        "displaylogo": False,
                        "scrollZoom": True,
                        "responsive": True,
                    },
                    style={
                        "height": "100%",
                        "width": "100%",
                    },
                )
            ],
            style={
                    "height": "60vh",
                    "width": "50%",
                    "minHeight": "350px",
                })
        ],
        style={
                "display": "flex",
                "width": "100%",
                "gap": "18px",
            })
    ]),
],
style={
        "backgroundColor": "#080c16",
        "color": "#e8edf5",
        "minHeight": "100vh",
        "padding": "24px 32px",
        "fontFamily": "Arial, sans-serif",
    }
)

app = Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([

    # Shared header
    header_layout,

    # Main navigation
    dcc.Tabs(
        id="main-view",
        value="explorer",
        children=[
            dcc.Tab(
                label="Aurora Database",
                value="explorer",
                style={
                    "borderRadius": "7px",
                    "backgroundColor": "#101827",
                    "border": "1px solid #34445c",
                    "color": "#8fa1b8",
                    "padding": "10px 22px",
                    "fontWeight": "bold",
                },
                selected_style={
                    "borderRadius": "7px",
                    "backgroundColor": "#168aad",
                    "border": "1px solid #2aa7c9",
                    "color": "#ffffff",
                    "padding": "10px 22px",
                    "fontWeight": "bold",
                },
            ),
            dcc.Tab(
                label="Spectrum Inspector",
                value="mlr",
                style={
                    "borderRadius": "7px",
                    "backgroundColor": "#101827",
                    "border": "1px solid #34445c",
                    "color": "#8fa1b8",
                    "padding": "10px 22px",
                    "fontWeight": "bold",
                },
                selected_style={
                    "borderRadius": "7px",
                    "backgroundColor": "#168aad",
                    "border": "1px solid #2aa7c9",
                    "color": "#ffffff",
                    "padding": "10px 22px",
                    "fontWeight": "bold",
                },
            ),
        ],
        style={
            "display": "flex",
            "alignItems": "end",
            "gap": "8px",
            "padding": "16px 20px",
            "backgroundColor": "#101827",
            "border": "1px solid #263348",
            "borderRadius": "10px",
            "marginBottom": "15px",
        },
    ),

    # Changes depending on selected tab
    html.Div(id="page-content"),

],
style={
        "backgroundColor": "#080c16",
        "color": "#e8edf5",
        "minHeight": "100vh",
        "padding": "24px 32px",
        "fontFamily": "Arial, sans-serif",
    }
)

@app.callback(
    Output("page-content", "children"),
    Input("main-view", "value"),
)
def switch_view(view):
    """Return the page layout selected by the top-level view control."""

    if view == "explorer":
        return explorer_layout

    elif view == "mlr":
        return mlr_layout

@app.callback(
    #Outputs: what get updated when a callback is initialized
    Output("map-image", "figure"),
    Output("globe-image", "figure"),
    Output("sza-image", "figure"),
    Output("lat-image", "figure"),
    Output("spectra-image", "figure"),
    Output("coverage-image", "figure"),
    Output("ls_ratio-image", "figure"),
    Output("lt-image", "figure"),
    Output("nightside-polar", "figure"),
    Output("nightside-cartesian", "figure"),
    Output("nightside-cartesian-mso", "figure"),
    #Inputs: what initializes a callback (e.g. clicking the load button)
    Input("load_button", "n_clicks"),
    #States: changes that only take effect AFTER the callback
    State("orb_start_input", "value"),
    State("orb_end_input", "value"),
    State("pearson", "value"),
    State("n_pix", "value"),
    State("sza_min", "value"),
    State("type-selector", "value"),
)

def update_figs(n_clicks, orb_start, orb_end, pearson, n_pix, sza_min, detection_type):
    """Filter the detection catalog and rebuild every Explorer figure.

    ``n_clicks`` is the Dash trigger; the remaining values are the filter state.
    The orbit interval is half-open: ``orb_start <= orbit < orb_end``.
    """

    # The arguments follow the Input/State order in the callback declaration.
    key=detection_type

    #Restrict by orbit selection
    filtered_data = data[
        (data["orbit_number"] >= orb_start) &
        (data["pearson_r_muv"] >= pearson) &
        (data["count"] >= n_pix) &
        (data["sza"] >= sza_min) &
        (data["orbit_number"] < orb_end)].copy()

    #Restrict by detection type
    if key == 'anom':
        filtered_data = filtered_data[filtered_data['CO/CO2p'] >= coUVD]
    if key == 'norm':
        filtered_data = filtered_data[filtered_data['CO/CO2p'] < coUVD]


    fig_map = plot_map(filtered_data, map_img, orb_start, orb_end, key)
    fig_globe = plot_sphere(filtered_data, map_img)
    fig_sza = sza_hist(filtered_data, key)
    fig_lat = latitude_hist(filtered_data, key)
    fig_coadd, wl, bg = coadd(filtered_data, key)
    fig_coverage, fig_ls_ratio = plot_coverage(filtered_data)
    fig_lt = plot_lt(filtered_data)
    fig_night, fig_night_cart, fig_night_cart_mso = plot_nightside(filtered_data)

    return fig_map, fig_globe, fig_sza, fig_lat, fig_coadd, fig_coverage, fig_ls_ratio, fig_lt, fig_night, fig_night_cart, fig_night_cart_mso


@app.callback(
    #Outputs: what get updated when a callback is initialized
    Output("spectrum-graph", "figure"),
    Output("detector_image", "figure"),
    Output("spectrum-information", "children"),
    Output("error-message", "children"),
    Output("cam-alt-graph", "figure"),
    Output("uvd-alt-graph", "figure"),
    #Inputs: what initializes a callback (e.g. clicking the load button)
    Input("load-button", "n_clicks"),
    Input("component-checklist", "value"),
    #States: changes that only take effect AFTER the callback
    State("orbit-input", "value"),
    State("scan-input", "value"),
    State("integration-input", "value"),
)
#Function that is run after the callback (updates spectrum after box tick or button press)
#NOTE: This is actually run at initialization with n_clicks=0, and just initializes a
def update_spectrum(
    n_clicks,
    selected_components,
    orbit_num,
    scan,
    integration,
):
    """Build the spectrum and detector figures for one selected integration.

    Scan and integration values come from the UI and are one-based. Any file,
    bounds, or FITS-format error is converted into a user-facing message so a
    bad selection does not terminate the Dash server.
    """
    #Error handling figure (safe mode)
    empty_figure = go.Figure()
    empty_figure_d = go.Figure()

    if orbit_num is None or scan is None or integration is None:
        return (
            empty_figure,
            empty_figure_d,
            "",
            "Enter an orbit, scan, and integration.",
            go.Figure(),
            go.Figure(),
        )
    #Get spectrum data and build figure based on inputs
    try:
        orbit_num = int(orbit_num)
        scan = int(scan)
        integration = int(integration)

        spectrum_data = build_spectrum(
            orbit_num,
            scan,
            integration,
        )

        detector_wavelength, detector_spectra, detector_log_spectra = build_detector_image(
            orbit_num,
            scan,
            integration,
        )

        cam, uvd, alt, rand_unc = get_alt_profiles(orbit_num, scan)

    #If error, set figure to empty and print error
    except Exception as error:
        return (
            empty_figure,
            empty_figure_d,
            "",
            f"Could not load spectrum: {error}",
            go.Figure(),
            go.Figure(),
        )

    #The initial figure that is loaded with the app (starts empty)
    fig = go.Figure()
    fig_detector = go.Figure()
    fig_alt_cam = go.Figure()
    fig_alt_uvd = go.Figure()

    wavelength = spectrum_data["wavelength"]

    #Build MLR figure
    for component in selected_components:
        #Skip un-checked components
        if component not in spectrum_data:
            continue
        #Add in the selected components
        fig.add_trace(
            go.Scatter(
                x=wavelength,
                y=spectrum_data[component],
                mode="lines",
                name=TRACE_NAMES[component],
                line=LINE_STYLES[component],
                hovertemplate=(
                    "Wavelength: %{x:.2f} nm"
                    "<br>DN: %{y:.3f}"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )

    #Build detector fig
    fig_detector.add_trace(
        go.Heatmap(
            z=detector_log_spectra,
            x=np.nanmean(detector_wavelength, axis=0),
            colorscale="Inferno",
            zmin=np.log10(50),
            zmax=np.log10(5000),
            colorbar=dict(showticklabels=False),
            showscale=False,
            hovertemplate="Wavelength: %{x:.2f} nm<br>DN: %{z:.3f}<extra></extra>",
        )
    )

    #Build alt profiles
    fig_alt_cam.add_trace(
        go.Scatter(
            x=cam,
            y=alt,
            mode="lines+markers",
            name="CO Cameron",
            hovertemplate=(
                "Brightness: %{x:.2f} kR"
                "<br>Altitude: %{y:.3f} km"
                "<extra>%{fullData.name}</extra>"
            ),
        )
    )

    fig_alt_uvd.add_trace(
        go.Scatter(
            x=uvd,
            y=alt,
            mode="lines+markers",
            name="CO2+ UVD",
            hovertemplate=(
                "Brightness: %{x:.2f} kR"
                "<br>Altitude: %{y:.3f} km"
                "<extra>%{fullData.name}</extra>"
            ),

        )
    )

    # Use the Cameron profile's brightness range for both altitude plots.
    finite_cam = np.asarray(cam)[np.isfinite(cam)]
    altitude_xaxis = {
        "title": "Brightness (kR)",
        "gridcolor": "#263348",
        "zerolinecolor": "#34445c",
        "linecolor": "#34445c",
        "tickcolor": "#34445c",
    }
    if finite_cam.size:
        cam_min = float(np.min(finite_cam))
        cam_max = float(np.max(finite_cam))
        cam_span = cam_max - cam_min
        x_padding = (
            0.05 * cam_span
            if cam_span > 0
            else max(abs(cam_max) * 0.05, 0.05)
        )
        altitude_xaxis["range"] = [cam_min - x_padding, cam_max + x_padding]

    # Match the altitude-profile styling to the MLR spectrum plot.
    for altitude_figure, title in (
        (fig_alt_cam, "CO Cameron Altitude Profile"),
        (fig_alt_uvd, "CO2+ UVD Altitude Profile"),
    ):
        altitude_figure.update_layout(
            title=title,
            xaxis=altitude_xaxis,
            yaxis={
                "title": "Altitude (km)",
                "gridcolor": "#263348",
                "zerolinecolor": "#34445c",
                "linecolor": "#34445c",
                "tickcolor": "#34445c",
            },
            hovermode="closest",
            paper_bgcolor="#080c16",
            plot_bgcolor="#101827",
            font={
                "color": "#c7d2e0",
                "family": "Arial, sans-serif",
            },
            height=500,
            showlegend=False,
            margin={
                "l": 70,
                "r": 30,
                "t": 60,
                "b": 70,
            },
            uirevision=f"{orbit_num}-{scan}",
        )
        altitude_figure.add_hline(
            y=135,
            line_width=1.5,
            line_dash="dash",
            line_color="#aeb8c6",
        )



    peaks = [193, 199.5, 206, 216, 228, 240, 289]
    for peak in peaks:
        fig_detector.add_vline(
            x=peak, 
            line_width=1, 
            line_dash="dash", 
            line_color="white"
        )

    #Load in labels and stuff
    fig.update_layout(
        title=(
            f"Orbit {orbit_num} — "
            f"Scan {scan}, Integration {integration}"
        ),
        xaxis={
            "title": "Wavelength (nm)",
            "gridcolor": "#263348",
            "zerolinecolor": "#34445c",
            "linecolor": "#34445c",
            "tickcolor": "#34445c",
        },
        yaxis={
            "title": "DN",
            "gridcolor": "#263348",
            "zerolinecolor": "#34445c",
            "linecolor": "#34445c",
            "tickcolor": "#34445c",
        },
        hovermode="x unified",
        paper_bgcolor="#080c16",
        plot_bgcolor="#101827",
        font={
            "color": "#c7d2e0",
            "family": "Arial, sans-serif",
        },
        height=750,
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.12,
            "xanchor": "center",
            "x": 0.5,
            "bgcolor": "rgba(0,0,0,0)",
        },
        margin={
            "l": 70,
            "r": 30,
            "t": 60,
            "b": 35,
        },
        uirevision=f"{orbit_num}-{scan}-{integration}",
    )

    #Load in labels and stuff for detector image
    fig_detector.update_layout(
        paper_bgcolor="#080c16",
        plot_bgcolor="#101827",
        font={
            "color": "#c7d2e0",
            "family": "Arial, sans-serif",
        },
        xaxis={
            "title": "Wavelength (nm)",
            "showticklabels": True,
            "gridcolor": "#263348",
            "linecolor": "#34445c",
            "tickcolor": "#34445c",
        },
        yaxis={
            "title": "Spatial Bin",
            "gridcolor": "#263348",
            "linecolor": "#34445c",
            "tickcolor": "#34445c",
        },
        height=160,
        margin={
            "l": 70,
            "r": 30,
            "t": 30,
            "b": 70,
        },
        uirevision=f"{orbit_num}-{scan}-{integration}",
    )

    #Calculate Pearson R
    r2, r = calculate_correlation(spectrum_data['wavelength'], spectrum_data['background'], spectrum_data['aurora'])

    #Bonus fun facts to be plotted (NOTE: check to see if threshold working)
    information = (
        f"Constant = {spectrum_data['constant_value']:.3f} DN | "
        f"Orbit threshold = "
        f"{spectrum_data['threshold_constant']:.3f} DN | "
        f"Pearson R: {r:.2f}"
    )

    return fig, fig_detector, information, "", fig_alt_cam, fig_alt_uvd

if __name__ == "__main__":
    app.run(debug=True)
