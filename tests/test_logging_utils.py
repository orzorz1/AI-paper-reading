from __future__ import annotations

import logging
import unittest

from paper_reading.logging_utils import LOGGER_NAME, configure_logging


class LoggingTests(unittest.TestCase):
    def test_configure_logging_sets_expected_level(self) -> None:
        logger = configure_logging(verbose=False)
        self.assertEqual(logger.name, LOGGER_NAME)
        self.assertEqual(logger.level, logging.INFO)
        self.assertGreaterEqual(len(logger.handlers), 1)

        verbose_logger = configure_logging(verbose=True)
        self.assertEqual(verbose_logger.level, logging.DEBUG)


if __name__ == "__main__":
    unittest.main()
