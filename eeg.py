from typing import List
import json
import mne
import pandas as pd
import serial.tools.list_ports
import brainflow.board_shim
from brainflow import BrainFlowInputParams, BoardShim, BoardIds
from nptyping import NDArray
import numpy as np
"""
version 00
"""
# This Message instructs the cyton dongle to configure electrodes gain as X6, and turn off last 3 electrodes
# NUM_CHANNELS_REMOVE = 3
HARDWARE_SETTINGS_MSG = "x1030110Xx2030110Xx3030110Xx4030110Xx5030110Xx6130110Xx7030110Xx8030110XxQ030110XxW130110XxE030110XxR130110XxT130110XxY131000XxU131000XxI131000X"
# STIM_CHAN_NAME = "Stim Markers"

HARDWARE_SETTINGS_MSG_old = "x1030110Xx2030110Xx3030110Xx4030110Xx5030110Xx6030110Xx7030110Xx8030110XxQ030110XxW030110XxE030110XxR030110XxT030110XxY131000XxU131000XxI131000X"


class Eeg:
    """
    This class is wraps all the communications with OpenBCI EEG helmet

    Attributes:
        board_id: int - id of the OpenBCI board
        ip_port: int - port for the board
        serial_port: str - serial port for the board
        headset: str - the headset name is used, it will be shown in the metadata
    """

    def __init__(self, board_id=BoardIds.CYTON_DAISY_BOARD, ip_port=6677, serial_port=None, headset='michael'):
        """
        Static parameters for configuring the static parameters
        and use it to wrap all the communications with the OpenBCI EEG
        All these static parameters will be in the __init__ function when we change it to a class
        """
        self.serial_port = serial_port
        # Board ID and headset name
        self.headset = headset  # string type
        self.board_id = board_id
        remove_channel = True

        # Set BrainFlow input parameters
        self.params = BrainFlowInputParams()
        self.ip_port = ip_port
        # params.serial_port = serial_port
        self.params.headset = headset
        self.params.board_id = board_id
        self.params.serial_port = serial_port if serial_port is not None else self.find_serial_port()
        # params.serial_port = find_serial_port()
        self.board = BoardShim(self.board_id, self.params)
        self.board.enable_dev_board_logger()
        # print(BoardShim.get_board_descr(board_id))  # Get board description
        print(json.dumps(self.board.get_board_descr(self.board_id), indent=4))  # Get pretty board description
        # Other Params
        self.sample_freq = self.board.get_sampling_rate(board_id)  # type int
        self.marker_row = self.board.get_marker_channel(board_id)  # type int
        self.eeg_names = self.get_board_names()

    def find_serial_port(self) -> str:
        """
        Return the string of the serial port to which the FTDI dongle is connected.
        If running in Synthetic mode, return ""
        Example: return "COM3"
        """
        if self.board_id == BoardIds.SYNTHETIC_BOARD:
            return ""
        else:
            plist = serial.tools.list_ports.comports()
            FTDIlist = [comport for comport in plist if comport.manufacturer == 'FTDI']
            if len(FTDIlist) > 1:
                raise LookupError(
                    "More than one FTDI-manufactured device is connected. Please enter serial_port manually.")
            if len(FTDIlist) < 1:
                raise LookupError("FTDI-manufactured device not found. Please check the dongle is connected")
            return FTDIlist[0].name

    def get_board_names(self) -> List[str]:
        """The method returns the board's channels"""
     #   if self.headset == 'yoav':
          #  return ["C3", "C4", "Cz", "FC1", "FC2", "FC5", "FC6", "CP1", "CP2", "CP5", "CP6", "O1", "O2"]
        return ["Pz","Fz","Cz","CP1","FC1","AF3","CP2","FC2","AF4"]
      #  else:
       #     return self.board.get_eeg_names(self.board_id)

    def get_board_channels(self, remove_channels) -> List[int]:
        """Get list with the channels locations as list of int"""
        if remove_channels:
        #    return self.board.get_eeg_channels(self.board_id)[:-3]
            return [1, 2, 3, 4, 5, 7, 8, 9, 11]
        else:
            return self.board.get_eeg_channels(self.board_id)

    def clear_board(self):
        """Clear all data from the EEG board"""

        # Get the data and don't save it
        self.board.get_board_data()

    def get_stream_data(self) -> NDArray:
        """The method returns the data from board and remove it"""
        return self.board.get_board_data()

    def get_channels_data(self, remove_channel) -> NDArray:
        """Get NDArray only with the channels data (without all the markers and other stuff)"""
        return self.board.get_board_data()[self.get_board_channels(remove_channel)]

    def stream_on(self):
        # Start board stream session
        # print(json.dumps(self.board.get_board_descr(self.board_id), indent=4))  # Get pretty board description
        self.board.prepare_session()
        self.board.config_board(HARDWARE_SETTINGS_MSG)
        self.board.start_stream()

    def stream_off(self):
        # time.sleep(10)  # time in seconds
        # Stop the board stream session
        self.board.log_message(brainflow.board_shim.LogLevels.LEVEL_INFO, "SAFE EXIT")
        self.board.stop_stream()
        self.board.release_session()
        # print(json.dumps(self.board.get_board_descr(self.board_id), indent=4))  # Get pretty board description

    def numpy_to_df(self, board_data: NDArray):
        """
        gets a Brainflow-style matrix and returns a Pandas Dataframe
        :param board_data: NDAarray retrieved from the board
        :returns df: a dataframe with the data
        """
        # create dictionary of <col index,col name> for renaming DF
        eeg_channels = self.board.get_eeg_channels(self.board_id)
        eeg_names = self.board.get_eeg_names(self.board_id)
        timestamp_channel = self.board.get_timestamp_channel(self.board_id)
        acceleration_channels = self.board.get_accel_channels(self.board_id)
        marker_channel = self.board.get_marker_channel(self.board_id)

        column_names = {}
        column_names.update(zip(eeg_channels, eeg_names))
        column_names.update(zip(acceleration_channels, ['X', 'Y', 'Z']))
        column_names.update({timestamp_channel: "timestamp", marker_channel: "marker"})

        df = pd.DataFrame(board_data.T)
        df.rename(columns=column_names)

        # drop unused channels
        df = df[column_names]

        # decode int markers
        # df['marker'] = df['marker'].apply(decode_marker)
        # df[['marker_status', 'marker_label', 'marker_index']] = pd.DataFrame(df['marker'].tolist(), index=df.index)
        return df

    def board_to_mne(self, board_data: NDArray, ch_names: List[str], sample_freq: int) -> mne.io.RawArray:
        """
        Convert the ndarray board data to mne object
        :param board_data: raw ndarray from board
        :return:
        """
        eeg_data = board_data / 1e6  # BrainFlow returns uV, convert to V for MNE

        # Creating MNE objects from BrainFlow data arrays
        ch_types = ['eeg'] * len(board_data)
        info = mne.create_info(ch_names=ch_names, sfreq=sample_freq, ch_types=ch_types)
        raw = mne.io.RawArray(eeg_data, info, verbose=False)

        return raw

    def insert_marker(self, status: str, label: int, index: int):
            """Insert an encoded marker into EEG data"""

            marker = self.encode_marker(status, label, index)  # encode marker
            self.board.insert_marker(marker)  # insert the marker to the stream

            # print(f'Status: { status }, Marker: { marker }')  # debug
            # print(f'Count: { self.board.get_board_data_count() }')  # debug

    def extract_trials(self, data: NDArray) -> [List[tuple], List[int]]:
        """
        The method get ndarray and extract the labels and durations from the data.
        :param data: the data from the board.
        :return:
        """

        # Init params
        durations, labels = [], []

        # Get marker indices
        markers_idx = np.where(data[self.marker_row, :] != 0)[0]

        # For each marker
        for idx in markers_idx:

            # Decode the marker
            status, label, _ = self.decode_marker(data[self.marker_row, idx])

            if status == 'start':

                labels.append(label)
                durations.append((idx,))

            elif status == 'stop':

                durations[-1] += (idx,)

        return durations, labels

    @staticmethod
    def filter_data(data, lowcut, highcut, sample_freq, notch_freq):
        y = mne.filter.notch_filter(data, Fs=sample_freq, freqs=notch_freq, verbose=False)
        d = mne.filter.filter_data(y, sample_freq, lowcut, highcut, verbose=False)
        return d

    @staticmethod
    def encode_marker(status: str, label: int, index: int):
        """
        Encode a marker for the EEG data.
        :param status: status of the stim (start/end)
        :param label: the label of the stim (Bibi -> 0, Sara -> 1, idle(Gantz) -> 2)
        :param index: index of the current label
        :return:
        """

        markerValue = 0
        if status == "start":
            markerValue += 1
        elif status == "stop":
            markerValue += 2
        else:
            raise ValueError("incorrect status value")

        markerValue += 10 * label

        markerValue += 100 * index

        return markerValue

    @staticmethod
    def decode_marker(marker_value: int) -> (str, int, int):
        """
        Decode the marker and return a tuple with the status, label and index.
        Look for the encoder docs for explanation for each argument in the marker.
        :param marker_value:
        :return:
        """
        if marker_value % 10 == 1:
            status = "start"
            marker_value -= 1
        elif marker_value % 10 == 2:
            status = "stop"
            marker_value -= 2
        else:
            raise ValueError("incorrect status value. Use start or stop.")

        label = ((marker_value % 100) - (marker_value % 10)) / 10

        index = (marker_value - (marker_value % 100)) / 100

        return status, int(label), int(index)

