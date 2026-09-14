"""Opt-in real physical learning profile using compact contraction scans."""

import hashlib
from pathlib import Path

import h01_contraction_event_profile as driver
from h01_contraction_scan import solve


def main():
    """Record the scan source identity and invoke the bounded profile driver.

    Returns
    -------
    None
        Writes synchronized profiles using the existing command arguments.
    """
    original_settings = driver.profile.numerical_settings

    def settings():
        result = original_settings()
        result['experimental_scan'] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(__file__).with_name('h01_contraction_scan.py'))}
        return result

    driver.profile.numerical_settings = settings
    driver.solve = solve
    driver.main()


if __name__ == '__main__':
    main()
