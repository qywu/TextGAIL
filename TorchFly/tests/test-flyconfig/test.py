import os
import sys
from typing import Any, Dict, List
import logging

from torchfly.flyconfig.flyconfig import check_valid_override, get_overrides


overrides = get_overrides(sys.argv[1:])