from os import getcwd, pardir, makedirs, path
from os import path
from os.path import join, abspath, exists

MAX_WORDS =20000
MAX_SEQUENCE_LENGTH = 600

ROOT_DIR = path.dirname(path.abspath(__file__)) # This is your Project Root

APP_DIR = join(ROOT_DIR, "app")

# ROOT level folders
DATA_DIR = join(ROOT_DIR, 'data')
if not exists(DATA_DIR):
    makedirs(DATA_DIR)

NOTEBOOKS_DIR = join(ROOT_DIR, 'notebookss')
if not exists(NOTEBOOKS_DIR):
    makedirs(NOTEBOOKS_DIR)

MODELS_DIR = join(ROOT_DIR, 'models')
if not exists(MODELS_DIR):
    makedirs(MODELS_DIR)

# DATA Level folders
FLAIR_DATA_DIR = join(DATA_DIR, "flair_data_dir")
if not exists(FLAIR_DATA_DIR):
    makedirs(FLAIR_DATA_DIR)

FLAIR_OUTPUT_DIR = join(DATA_DIR, "flair_output_dir")
if not exists(FLAIR_OUTPUT_DIR):
    makedirs(FLAIR_OUTPUT_DIR)

FLAIR_EMDG_DIR = join(DATA_DIR, "flair_emdg_dir")
if not exists(FLAIR_EMDG_DIR):
    makedirs(FLAIR_EMDG_DIR)

# NOTEBOOK Level folders
TALOS_DIR = join(NOTEBOOKS_DIR, 'talos_logs')
if not exists(TALOS_DIR):
    makedirs(TALOS_DIR)