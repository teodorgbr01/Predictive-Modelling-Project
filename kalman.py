"""Simple 1D Kalman filter utilities.

This module provides:
- A reusable scalar Kalman filter class.
- A helper to filter a full sequence of measurements.
- A small demo when running this file directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import gauss, seed
from typing import Iterable, List


@dataclass
class ScalarKalmanFilter:
	"""A 1D Kalman filter for noisy scalar measurements.

	Attributes:
		x: Current state estimate.
		p: Current estimate covariance (uncertainty).
		q: Process noise covariance.
		r: Measurement noise covariance.
	"""

	x: float = 0.0
	p: float = 1.0
	q: float = 0.01
	r: float = 1.0

	def predict(self) -> float:
		"""Prediction step for a constant-state model."""
		self.p = self.p + self.q
		return self.x

	def update(self, z: float) -> float:
		"""Correction step with a new measurement."""
		k = self.p / (self.p + self.r)
		self.x = self.x + k * (z - self.x)
		self.p = (1.0 - k) * self.p
		return self.x

	def step(self, z: float) -> float:
		"""Run predict + update and return the filtered value."""
		self.predict()
		return self.update(z)


def apply_kalman_filter(
	measurements: Iterable[float],
	initial_estimate: float = 0.0,
	initial_error: float = 1.0,
	process_noise: float = 0.01,
	measurement_noise: float = 1.0,
) -> List[float]:
	"""Filter a sequence of measurements using a scalar Kalman filter."""
	kf = ScalarKalmanFilter(
		x=initial_estimate,
		p=initial_error,
		q=process_noise,
		r=measurement_noise,
	)

	filtered: List[float] = []
	for value in measurements:
		filtered.append(kf.step(float(value)))
	return filtered


def _demo() -> None:
	"""Run a tiny demo with synthetic noisy measurements."""
	seed(42)
	true_value = 10.0
	noisy_measurements = [true_value + gauss(0, 2) for _ in range(12)]

	filtered_values = apply_kalman_filter(
		noisy_measurements,
		initial_estimate=0.0,
		initial_error=4.0,
		process_noise=0.05,
		measurement_noise=4.0,
	)

	print("index\tmeasurement\tfiltered")
	for idx, (measurement, estimate) in enumerate(
		zip(noisy_measurements, filtered_values),
		start=1,
	):
		print(f"{idx:>2}\t{measurement:>11.4f}\t{estimate:>8.4f}")


if __name__ == "__main__":
	_demo()
