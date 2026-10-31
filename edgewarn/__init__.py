"""Package shim: the modules sit in the repository root; expose them as edgewarn.*."""
import os
__path__.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
