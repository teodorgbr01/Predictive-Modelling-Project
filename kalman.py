"""Simple 1D Kalman filter utilities.

This module provides:
- A reusable scalar Kalman filter class.
- A helper to filter a full sequence of measurements.
- A small demo when running this file directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import gauss, seed
from typing import Iterable, List, Optional


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

	def __post_init__(self) -> None:
		"""Validate noise/covariance values after initialization."""
		if self.p <= 0:
			raise ValueError("initial covariance p must be > 0")
		if self.q < 0:
			raise ValueError("process noise q must be >= 0")
		if self.r <= 0:
			raise ValueError("measurement noise r must be > 0")

	def predict(self) -> float:
		"""Prediction step for a constant-state model."""
		self.p = self.p + self.q
		return self.x

	def update(self, z: float, measurement_noise: Optional[float] = None) -> float:
		"""Correction step with a new measurement."""
		r_value = self.r if measurement_noise is None else float(measurement_noise)
		if r_value <= 0:
			raise ValueError("measurement_noise must be > 0")

		k = self.p / (self.p + r_value)
		self.x = self.x + k * (z - self.x)
		self.p = (1.0 - k) * self.p
		return self.x

	def step(self, z: float, measurement_noise: Optional[float] = None) -> float:
		"""Run predict + update and return the filtered value."""
		self.predict()
		return self.update(z, measurement_noise=measurement_noise)


def apply_kalman_filter(
	measurements: Iterable[float],
	initial_estimate: float = 0.0,
	initial_error: float = 1.0,
	process_noise: float = 0.01,
	measurement_noise: float = 1.0,
	measurement_noises: Optional[Iterable[float]] = None,
) -> List[float]:
	"""Filter a sequence of measurements using a scalar Kalman filter.

	Args:
		measurements: Input measurements.
		initial_estimate: Initial state estimate.
		initial_error: Initial covariance.
		process_noise: Process noise covariance.
		measurement_noise: Default measurement noise covariance.
		measurement_noises: Optional per-measurement noise values.
	"""
	kf = ScalarKalmanFilter(
		x=initial_estimate,
		p=initial_error,
		q=process_noise,
		r=measurement_noise,
	)

	noise_iter = iter(measurement_noises) if measurement_noises is not None else None
	filtered: List[float] = []
	for value in measurements:
		if noise_iter is None:
			current_noise = None
		else:
			try:
				current_noise = next(noise_iter)
			except StopIteration:
				current_noise = None
		filtered.append(kf.step(float(value), measurement_noise=current_noise))
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
