import streamlit as st
import os

from torch.utils.data import DataLoader
from config import get_config_universal
from dataset import DataSet
from datasetbuilder import DataSetBuilder
from test import Test
from utils.pytorchtools import load_trained_model
from visualization.steamlit_plot import plot_kinematic_predictions


dataset_name = 'camargo'
config = get_config_universal(dataset_name)

# model_file = 'transformertsai_g1g2rardsasd_g1g2rardsasd.pt'
# model = torch.load(os.path.join('./caches/trained_model/v05', model_file))
sensor_options = {'Thigh & Shank & Foot': ['foot', 'shank', 'thigh'],
                  'Thigh & Shank': ['thigh', 'shank'],
                  'Thigh & Foot': ['thigh', 'foot'],
                  'Shank & Foot': ['shank', 'foot'],
                  'Thigh': ['thigh'],
                  'Shank': ['shank'],
                  'Foot': ['foot']}
model_names = {'CNNLSTM': 'hernandez2021cnnlstm', 'BiLSTM': 'bilstm', 'BioMAT': 'transformertsai'}
sensor_file_keys = {
    'Thigh & Shank & Foot': 'thighshankfoot',
    'Thigh & Shank': 'thighshank',
    'Thigh & Foot': 'thighfoot',
    'Shank & Foot': 'shankfoot',
    'Thigh': 'thigh',
    'Shank': 'shank',
    'Foot': 'foot',
}


def resolve_model_path(sensor_name, model_name):
    sensor_key = sensor_file_keys[sensor_name]
    stem = model_names[model_name]
    if sensor_key == 'thighshankfoot':
        model_file = f'{stem}_g1g2rardsasd_g1g2rardsasd.pt'
    else:
        model_file = f'{sensor_key}_{stem}_g1g2rardsasd_g1g2rardsasd.pt'
    return os.path.join('./caches/trained_model/', model_file)


def apply_model_contract(config, model, model_name):
    """Align data config to the loaded model (seq_len, optional checkpoint metadata)."""
    config['model_name'] = model_names[model_name]
    seq_len = getattr(model, 'seq_len', None)
    if seq_len is None and hasattr(model, 'W_pos'):
        seq_len = int(model.W_pos.shape[0])
    if seq_len is not None:
        config['target_padding_length'] = int(seq_len)

    meta = getattr(model, 'checkpoint_meta', None) or {}
    if meta.get('window_length'):
        config['target_padding_length'] = int(meta['window_length'])
    if meta.get('sites'):
        config['selected_sensors'] = list(meta['sites'])
    if meta.get('targets'):
        config['selected_opensim_labels'] = list(meta['targets'])
    return config, meta

@st.cache_data
def fetch_data(config):
    dataset_handler = DataSet(config, load_dataset=True)
    kihadataset_train, kihadataset_test = dataset_handler.run_dataset_split_loop()
    kihadataset_train['x'], kihadataset_train['y'], kihadataset_train['labels'] = dataset_handler.run_segmentation(
        kihadataset_train['x'],
        kihadataset_train['y'], kihadataset_train['labels'])
    kihadataset_test['x'], kihadataset_test['y'], kihadataset_test['labels'] = dataset_handler.run_segmentation(
        kihadataset_test['x'],
        kihadataset_test['y'], kihadataset_test['labels'])
    train_dataset = DataSetBuilder(kihadataset_train['x'], kihadataset_train['y'], kihadataset_train['labels'],
                                   transform_method=config['data_transformer'], scaler=None)
    test_dataset = DataSetBuilder(kihadataset_test['x'], kihadataset_test['y'], kihadataset_test['labels'],
                                  transform_method=config['data_transformer'], scaler=train_dataset.scaler)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=False)
    return test_dataloader, kihadataset_test

# @st.cache_data
def fetch_model(model_path):
    return load_trained_model(model_path)

# @st.cache_data
def fetch_predictions(model):
    test_handler = Test()
    y_pred, y_true, loss = test_handler.run_testing(config, model, test_dataloader=test_dataloader)
    y_true = y_true.detach().cpu().clone().numpy()
    y_pred = y_pred.detach().cpu().clone().numpy()
    return y_pred, y_true, loss

st.set_page_config(layout="wide")
st.title('BioMAT: An Open-Source Biomechanics Multi-Activity Transformer for Joint Kinematic Predictions using Wearable Sensors')

st.sidebar.title('Sensor Configuration')
selected_sensor = st.sidebar.selectbox('Pick one', ['Thigh & Shank & Foot',
                                                    'Thigh & Shank',
                                                    'Thigh & Foot',
                                                    'Shank & Foot',
                                                    'Thigh',
                                                    'Shank',
                                                    'Foot'])

config['selected_sensors'] = sensor_options[selected_sensor]

st.sidebar.title('Model Configuration')
selected_model = st.sidebar.selectbox('Pick one', ['CNNLSTM',
                                                    'BiLSTM',
                                                   'BioMAT'])

st.sidebar.title('Subject')
selected_subject = st.sidebar.slider('Pick a Subject Number', min_value=23, max_value=25, step=1)

st.sidebar.title('Activity')
selected_activities = st.sidebar.multiselect('Pick Output Activities',
                                           ['LevelGround Walking', 'Ramp Ascent', 'Ramp Descent', 'Stair Ascent', 'Stair Descent'])

index_to_plot = st.sidebar.number_input(
    'Starting window index (plots 3 windows)',
    min_value=0,
    max_value=5,
    value=0,
)

if st.sidebar.button('Predict'):
    model_path = resolve_model_path(selected_sensor, selected_model)
    if not os.path.isfile(model_path):
        st.error(f'Model file not found: {model_path}')
        st.stop()

    with st.spinner('Model is loading...'):
        model = fetch_model(model_path)
    config, checkpoint_meta = apply_model_contract(config, model, selected_model)
    st.info(
        f"Using model contract: sensors={config['selected_sensors']}, "
        f"window={config['target_padding_length']}"
    )
    st.success('Model is loaded!')

    with st.spinner('Data is loading...'):
        test_dataloader, kihadataset_test = fetch_data(config)
    st.success('Data is loaded!')

    with st.spinner('Prediction ...'):
        y_pred, y_true, loss = fetch_predictions(model)
    st.success('Prediction is Completed!')
    subject = 'AB' + str(selected_subject)
    fig = plot_kinematic_predictions(
        y_true,
        y_pred,
        kihadataset_test['labels'],
        subject,
        selected_activities=selected_activities,
        selected_index_to_plot=index_to_plot,
        n_windows=3,
    )
    st.plotly_chart(fig, use_container_width=True)



