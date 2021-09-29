import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import pickle
from PyQt5 import QtCore, QtWidgets, QtGui
from interactive_filter_window import Ui_MainWindow
# from arctools.dbconnect import DBconnect
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.pyplot import Figure
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn import svm
from sklearn.covariance import EllipticEnvelope


class MainApp(Ui_MainWindow):
    def __init__(self, parent):
        self.setupUi(parent)
        parent.setWindowTitle('Interactive Filter')

        # get the range of changing ATM contracts
        self.symbols = ['XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY']
        self.full_atm_etfs_implied_vol_list = self.load_data_from_pickle('march_contract_implied_vol_atm_etf_dfs')
        self.atm_strike_range_dict = {}
        self.get_atm_strike_range_for_each_asset()

        # connect buttons
        self.lock_asset_button.clicked.connect(self.lock_button_action)
        self.filter_button.clicked.connect(self.filter_action)
        self.save.clicked.connect(self.save_action)
        self.robust_covariance_button.clicked.connect(lambda: self.select_one_filter_result('robust_covariance_df'))
        self.one_class_svm_button.clicked.connect(lambda: self.select_one_filter_result('one_class_df'))
        self.isolation_forest_button.clicked.connect(lambda: self.select_one_filter_result('isolation_forest_df'))
        self.local_outlier_factor_button.clicked.connect(lambda: self.select_one_filter_result('local_outlier_factor_df'))

        # connect both serve database and local database
        # conn = DBconnect()
        # conn.read_config('config.ini')
        # self.engine = conn.connections['data'].ENGINE
        # self.local_engine = conn.connections['local_clean_data'].ENGINE
        # self.data_base_conn = self.engine.connect()

        # copy the req_contract table from serve to local, run once
        # self.copy_the_req_contract_table()

        # set up filters
        self.filter_methods_dict = {'Robust covariance': EllipticEnvelope(contamination=0.03),
                                    'One-Class SVM': svm.OneClassSVM(nu=0.03, kernel='rbf', gamma=0.1),
                                    'Isolation Forest': IsolationForest(contamination=0.03, random_state=42),
                                    'Local Outlier Factor': LocalOutlierFactor(n_neighbors=35, contamination=0.03)}

        # some class variables
        self.asset_symbol = None
        self.asset_expiry = None
        self.range = None
        self.pnl_graph = None

        # info about the contact being filtered
        self.strike_index = 0
        self.option_type = None
        self.current_contract = None
        self.filtered_contract = None

        # filtered df using different method
        self.robust_covariance_df = None
        self.one_class_df = None
        self.isolation_forest_df = None
        self.local_outlier_factor_df = None

        # pop out windows to contain the graph of result of each method
        self.background_robust_covariance = None
        self.background_one_class_svm = None
        self.background_isolation_forest = None
        self.background_local_outlier_factor = None

    @ staticmethod
    def load_data_from_pickle(file_name):
        file_object = open(file_name, 'rb')
        df = pickle.load(file_object)
        file_object.close()

        return df

    # get changing ATM contracts for each asset
    def get_atm_strike_range_for_each_asset(self):
        for symbol, df_temp in zip(self.symbols, self.full_atm_etfs_implied_vol_list):
            call_strike = set(df_temp.sort_values(by='call_strike')['call_strike'])
            put_strike = set(df_temp.sort_values(by='put_strike')['put_strike'])
            self.atm_strike_range_dict[symbol] = [call_strike, put_strike]

    # save the selected result and print on log
    def select_one_filter_result(self, method_to_choose):
        output = 'You select the filter of {} to process contract {} {} expires at {} with strike {}'.format(
            method_to_choose, self.asset_symbol, self.option_type, self.asset_expiry, self.range[self.strike_index]
        )
        self.filtered_contract = self.__getattribute__(method_to_choose)
        self.textBrowser.append(output)

    # set up pop up window for graph of each result
    def set_up_background_for_graph(self):
        self.background_robust_covariance = QtWidgets.QWidget()
        self.background_robust_covariance.setWindowTitle('Robust covariance')
        self.background_robust_covariance.setGeometry(QtCore.QRect(10, 250, 1200, 450))

        self.background_one_class_svm = QtWidgets.QWidget()
        self.background_one_class_svm.setWindowTitle('One-Class SVM')
        self.background_one_class_svm.setGeometry(QtCore.QRect(10, 250, 1200, 450))

        self.background_isolation_forest = QtWidgets.QWidget()
        self.background_isolation_forest.setWindowTitle('Isolation Forest')
        self.background_isolation_forest.setGeometry(QtCore.QRect(10, 250, 1200, 450))

        self.background_local_outlier_factor = QtWidgets.QWidget()
        self.background_local_outlier_factor.setWindowTitle('Local Outlier Factor')
        self.background_local_outlier_factor.setGeometry(QtCore.QRect(10, 250, 1200, 450))

    # plot each result on corresponding pop up window
    def plot_result_of_each_method(self, df_before):
        self.graph_method_1 = GraphWidget()
        self.graph_method_2 = GraphWidget()
        self.graph_method_3 = GraphWidget()
        self.graph_method_4 = GraphWidget()

        self.plot_one_result(self.graph_method_1, df_before, self.robust_covariance_df,
                             self.background_robust_covariance)
        self.plot_one_result(self.graph_method_2, df_before, self.one_class_df,
                             self.background_one_class_svm)
        self.plot_one_result(self.graph_method_3, df_before, self.isolation_forest_df,
                             self.background_isolation_forest)
        self.plot_one_result(self.graph_method_4, df_before, self.local_outlier_factor_df,
                             self.background_local_outlier_factor)

    # plot function
    @ staticmethod
    def plot_one_result(graph_object, df_before, df, background):
        layout = QtWidgets.QVBoxLayout(graph_object)
        figure = Figure(figsize=(5, 5), dpi=100)
        graph_object.canvas = FigureCanvasQTAgg(figure)
        toolbar = NavigationToolbar(graph_object.canvas, graph_object)
        layout.addWidget(toolbar)
        layout.addWidget(graph_object.canvas)
        graph_object.setLayout(layout)
        graph_object.plot(df_before['option_close'], df['option_close'])
        background.setLayout(layout)
        background.show()

    # copy the req contract table from serve database
    def copy_the_req_contract_table(self):
        sql = """
        select *
        from req_contracts
        where (m_symbol like 'X%%' or m_symbol = 'SPX')
        """
        df = pd.read_sql(sql, self.data_base_conn)
        local_conn = self.local_engine.connect()
        df.to_sql('req_contracts', con=local_conn, if_exists='append', index=False)
        local_conn.close()

    # do filter
    def filter_action(self):
        # set up pop up windows
        self.set_up_background_for_graph()

        # todo: write the first instance at the beginning
        # if this is the first run of this contract
        if self.filtered_contract is not None:
            # do filter using all four methods
            self.do_filter_on_four_methods(self.filtered_contract)

            # plot the filtered results
            self.plot_result_of_each_method(self.filtered_contract)

        # if this is a filter again operation
        else:
            # do filter using all four methods
            self.do_filter_on_four_methods(self.current_contract)

            # plot the filtered results
            self.plot_result_of_each_method(self.current_contract)

    # get filtered data drame using all four methods
    def do_filter_on_four_methods(self, df):
        self.robust_covariance_df = self.clean_data_process(df, 'Robust covariance', 'time_to_maturity',
                                                            'intrinsic_value', 'option_close')
        self.one_class_df = self.clean_data_process(df, 'One-Class SVM', 'time_to_maturity',
                                                    'intrinsic_value', 'option_close')
        self.isolation_forest_df = self.clean_data_process(df, 'Isolation Forest', 'time_to_maturity',
                                                           'intrinsic_value', 'option_close')
        self.local_outlier_factor_df = self.clean_data_process(df, 'Local Outlier Factor',
                                                               'time_to_maturity', 'intrinsic_value', 'option_close')

    def clean_data_process(self, df, method_to_use, x_axis, x2_axis, y_axis):
        # get data set
        X = df.loc[:, [x_axis, x2_axis, y_axis]]

        # todo: 1) if method_to_use in self.filter_method, do the line, else: raise an error (method to use is not in
        # todo: 2) use assert statement:
        # get filter method
        classifier = self.filter_methods_dict[method_to_use]

        # plot data before the filter
        plt.figure('{}'.format(method_to_use))
        df[y_axis].plot()

        # do filter
        if method_to_use == 'Local Outlier Factor':
            y_pred = classifier.fit_predict(X)
        else:
            y_pred = classifier.fit(X).predict(X)

        df['outlier_flag'] = y_pred
        new_df = df[df['outlier_flag'] == 1]

        # plot data after the filter
        new_df[y_axis].plot()

        return new_df

    # to lock on the contracts selected and import the first one from serve database
    def lock_button_action(self):
        # get symbol, expiry, and strike range
        self.asset_symbol = self.asset_symbol_combobox.currentText()
        self.asset_expiry = self.asset_expiry_combobox.currentText()

        # todo: assert those are numbers
        lower_bound = self.strike_lower_bond_lineEdit.text()
        upper_bound = self.strike_upper_bound_lineEdit.text()
        
        self.range = np.arange(int(lower_bound), int(upper_bound))

        # get the first call contract of selected option chain
        self.option_type = 'C'
        self.current_contract = self.get_single_strike_etf_option_from_serve(self.asset_expiry, self.asset_symbol,
                                                                             self.range[self.strike_index],
                                                                             self.option_type)

        # show current processing contract in main window
        self.current_contract_show_label.setText('{} {} expires at {} with strike {}'.format(
            self.asset_symbol, self.option_type, self.asset_expiry, self.range[self.strike_index]))

    # get selected single contract from database
    def get_single_strike_etf_option_from_serve(self, expiry_date, symbol_name, strike, option_type):
        sql = """
        SELECT a.symbol_id as underlying_id, c.m_sectype as underlying_type, d.id as option_id, 
        d.m_expiry as option_expiry, d.m_sectype as option_sectype, d.m_right as option_right, 
        c.m_localsymbol as underlying_symbol,d.m_localsymbol,a.date,a.close as underlying,
        b.close as option_close, b.high as option_high, b.low as option_low, d.m_strike as strike
        FROM data_ib_equity_1min a, data_ib_options_midpoint_3min b, req_contracts c, req_contracts d
        WHERE a.symbol_id = c.id
        AND c.m_sectype = 'STK'
        AND c.m_symbol = '{}'
        AND b.symbol_id = d.id
        AND d.m_strike = '{}'
        AND d.m_expiry::date = '{}'
        AND d.m_sectype = 'OPT' AND d.m_symbol = '{}' AND d.m_right = '{}'                                                           
        AND a.date = b.date
        AND a.date::date >= '2018-10-23' AND a.date::date <= '2019-01-25'                                                                
        AND b.date::date >= '2018-10-23' AND b.date::date <= '2019-01-25'
        ORDER BY a.date, d.m_strike DESC;
        """.format(symbol_name, strike, expiry_date, symbol_name, option_type)
        print(sql)
        df = pd.read_sql(sql, self.data_base_conn)
        df = df.sort_values(by=['date', 'strike'])
        df['time_to_maturity'] = (dt.datetime(2019, 3, 15) - df['date']) / dt.timedelta(days=365)
        if option_type == 'C':
            df['intrinsic_value'] = df['underlying'] - df['strike']
        else:
            df['intrinsic_value'] = df['strike'] - df['underlying']

        df = df.set_index('date')
        # df = df.resample('3T').first().dropna()

        return df

    # save filtered data to local database
    def save_to_database(self):

        self.filtered_contract['date'] = self.filtered_contract.index
        self.filtered_contract['null'] = float('NaN')
        self.filtered_contract['zeros'] = 0

        underlying_data = self.filtered_contract[['date', 'underlying_id', 'zeros', 'zeros', 'zeros',
                                                  'underlying', 'zeros']]
        underlying_data.columns = ['date', 'symbol_id', 'open', 'high', 'low', 'close', 'volume']

        option_data = self.filtered_contract[['date', 'option_id', 'option_high', 'option_low', 'option_close']]
        option_data.columns = ['date', 'symbol_id', 'high', 'low', 'close']
        
        underlying_data.to_sql('data_ib_equity_1min', con=self.local_engine, if_exists='append', index=False)

        option_data.to_sql('data_ib_options_midpoint_3min', con=self.local_engine, if_exists='append', index=False)

    def save_action(self):
        # save filtered data to local database
        self.save_to_database()

        # print str in log window
        output = 'You saved the contract to local data base'
        format_output = output.center(90, '*')

        self.textBrowser.setTextColor(QtGui.QColor(255, 0, 0))
        self.textBrowser.append(format_output)
        self.textBrowser.setTextColor(QtGui.QColor(0, 0, 0))

        self.filtered_contract = None

        # Change current contract
        if self.option_type == 'C':
            self.option_type = 'P'

        elif self.option_type == 'P':

            if self.strike_index == len(self.range):
                print('This asset symbol is done!')

            else:
                self.strike_index += 1
                self.option_type = 'C'

        # todo: if the contract is done, do not load new data
        if self.strike_index == len(self.range) and self.option_type == 'P':
            print('This asset symbol is done!')

        # get call contract with next strike in strike range list
        self.current_contract = self.get_single_strike_etf_option_from_serve(self.asset_expiry, self.asset_symbol,
                                                                             self.range[self.strike_index],
                                                                             self.option_type)

        # change the current processing label
        self.current_contract_show_label.setText('{} {} expires at {} with strike {}'.format(
            self.asset_symbol, self.option_type, self.asset_expiry, self.range[self.strike_index]))


# Graph Widget class for plot graphs
class GraphWidget(QtWidgets.QWidget):
    def __init__(self, parent_ =None):
        super().__init__(parent_)
        self.setGeometry(QtCore.QRect(10, 60, 1000, 530))
        self.setObjectName('graph_widget')
        self.canvas = None

    def setup_figures(self):
        layout = QtWidgets.QVBoxLayout(self)
        figure = Figure(figsize=(5, 5), dpi=100)
        self.canvas = FigureCanvasQTAgg(figure)
        toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def start_drawing(self):
        self.drawer = QtCore.QTimer()
        self.drawer.setInterval(50000)
        self.drawer.timeout.connect(self.draw_figure)
        self.drawer.start()

    def plot(self, data_, data_2):
        # self.canvas.draw_new(data_)
        self.canvas.ax = self.canvas.figure.subplots()
        self.canvas.ax.clear()
        data_ = data_.sort_index()
        data_2 = data_2.sort_index()
        self.canvas.ax.plot(data_2)
        self.canvas.ax.plot(data_)
        self.canvas.figure.autofmt_xdate()
        self.canvas.draw()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = MainApp(MainWindow)
    MainWindow.show()

    sys.exit(app.exec_())
