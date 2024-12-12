from AlgorithmImports import *
import numpy as np
from scipy.optimize import root

class PairsTradingGBM(QCAlgorithm):

    def Initialize(self):
        # Set the starting and ending date for backtest
        self.SetStartDate(2010, 1, 1)
        self.SetEndDate(2024, 1, 1)

        # Set initial capital
        self.SetCash(100000)

        # Add the stocks to the universe
        self.msft = self.AddEquity("WMT", Resolution.Daily).Symbol
        self.aapl = self.AddEquity("TGT", Resolution.Daily).Symbol

        # Warm-up period for getting historical data
        self.SetWarmup(252)

        # Placeholder for historical data and threshold values
        self.history = None
        self.k1, self.k2, self.k3 = None, None, None

        # Schedule to check daily
        self.Schedule.On(
            self.DateRules.EveryDay(), self.TimeRules.At(10, 0), self.TradePairs
        )

    def OnData(self, data):
        # Collect historical data during warm-up
        if self.IsWarmingUp:
            return

        # Fetch historical data if not done yet
        if self.history is None:
            self.Debug("Fetching historical data...")
            self.history = self.History([self.msft, self.aapl], 252, Resolution.Daily)
            if self.history.empty:
                self.Debug("Historical data is empty")
                return
            self.CalculateThresholds()

    def CalculateThresholds(self):
        # Check if history is not empty
        if self.history.empty:
            self.Debug("Historical data is empty in CalculateThresholds")
            return

        # Use the MultiIndex to get the price series
        try:
            msft_history = self.history.loc[self.msft]
            aapl_history = self.history.loc[self.aapl]
        except KeyError as e:
            self.Debug(f"KeyError in history data: {e}")
            return

        # Ensure we have sufficient data
        if msft_history.empty or aapl_history.empty:
            self.Debug("MSFT or AAPL historical data is empty")
            return

        msft_prices = msft_history["close"].values
        aapl_prices = aapl_history["close"].values

        # Ensure the prices arrays are of the same length
        min_length = min(len(msft_prices), len(aapl_prices))
        msft_prices = msft_prices[-min_length:]
        aapl_prices = aapl_prices[-min_length:]

        # Calculate daily returns
        msft_returns = np.diff(np.log(msft_prices))
        aapl_returns = np.diff(np.log(aapl_prices))

        # Calculate drift and volatility (daily)
        mu1 = np.mean(msft_returns)
        mu2 = np.mean(aapl_returns)
        sigma11 = np.std(msft_returns)
        sigma22 = np.std(aapl_returns)
        rho = np.corrcoef(msft_returns, aapl_returns)[0, 1]

        # Define constants for transaction costs and discount rate
        beta_b, beta_s, discount_rate = 1.001, 0.999, 0.05

        # Calculate thresholds
        self.Debug("Calculating thresholds...")
        self.k1, self.k2, self.k3 = self.calculate_thresholds(
            mu1, mu2, sigma11, sigma22, rho, beta_b, beta_s, discount_rate
        )

    def TradePairs(self):
        self.Log("TradePairs method called.")
        # Ensure thresholds are calculated
        if self.k1 is None or self.k2 is None or self.k3 is None:
            self.Debug("Thresholds not calculated yet")
            return

        msft_price = self.Securities[self.msft].Price
        aapl_price = self.Securities[self.aapl].Price
        spread = msft_price - aapl_price

        self.Debug(f"Spread: {spread}, k1: {self.k1}, k2: {self.k2}, k3: {self.k3}")

        # Buy condition (spread below k1)
        if spread < self.k1 and not self.Portfolio[self.msft].Invested:
            self.SetHoldings(self.msft, 0.5)
            self.SetHoldings(self.aapl, -0.5)

        # Sell condition (spread above k2)
        elif spread > self.k2 and self.Portfolio[self.msft].Invested:
            self.Liquidate()

        # Stop-loss condition (spread above k3)
        elif spread > self.k3 and self.Portfolio[self.msft].Invested:
            self.Liquidate()

    def calculate_thresholds(
        self, mu1, mu2, sigma11, sigma22, rho, beta_b, beta_s, discount_rate
    ):
        # Calculate intermediate values
        self.Debug("Calculating thresholds...")
        a11 = sigma11**2 + rho * sigma11 * sigma22
        a12 = sigma11 * sigma22 * (1 + rho)
        a22 = sigma22**2 + rho * sigma11 * sigma22
        lambda_ = (a11 - 2 * a12 + a22) / 2

        # Check if lambda_ is valid
        if lambda_ == 0:
            self.Debug("Lambda is zero, cannot calculate thresholds")
            return None, None, None

        # Calculate delta values
        discriminant = (1 + (mu1 - mu2) / lambda_) ** 2 + 4 * discount_rate / lambda_
        if discriminant < 0:
            self.Debug("Discriminant is negative, cannot calculate square root")
            return None, None, None
        sqrt_discriminant = np.sqrt(discriminant)

        delta_1 = 0.5 * (1 + (mu1 - mu2) / lambda_ + sqrt_discriminant)
        delta_2 = 0.5 * (1 + (mu1 - mu2) / lambda_ - sqrt_discriminant)

        # Find r0
        def f(r):
            r = r[0]
            term1 = (
                delta_1
                * (1 - delta_2)
                * (beta_b * r**-delta_2 - beta_s)
                * (beta_b - beta_s * r ** (1 - delta_1))
            )
            term2 = (
                delta_2
                * (1 - delta_1)
                * (beta_b * r**-delta_1 - beta_s)
                * (beta_b - beta_s * r ** (1 - delta_2))
            )
            return term1 - term2

        r0_guess = (beta_b / beta_s) ** 2
        result = root(f, [r0_guess], method="hybr")

        if not result.success:
            self.Debug(f"Root finding did not converge: {result.message}")
            return None, None, None
        r0 = result.x[0]

        # Calculate k1 and k2
        k1 = delta_2 * (beta_b * r0**-delta_1 - beta_s) / ((1 - delta_2) * (beta_b - beta_s * r0 ** (1 - delta_1)))
        k2 = delta_1 * (beta_b * r0**-delta_2 - beta_s) / ((1 - delta_1) * (beta_b - beta_s * r0 ** (1 - delta_2)))

        # Solve for k3
        def f2(x):
            term1 = (x * (1 - delta_2) * beta_s + delta_2 * beta_b) / x**delta_1
            term2 = ((delta_1 - 1) * beta_s - delta_1 * beta_b) / x**delta_2
            term3 = (beta_s + beta_b) * (delta_1 - delta_2)
            return term1 + term2 + term3

        k3_guess = (k1 + k2) / 2
        result = root(f2, [k3_guess], method="hybr")

        if not result.success:
            self.Debug(f"Root finding for k3 did not converge: {result.message}")
            return k1, k2, None

        k3 = result.x[0]
        self.Debug(f"k1: {k1}, k2: {k2}, k3: {k3}")
        return k1, k2, k3
