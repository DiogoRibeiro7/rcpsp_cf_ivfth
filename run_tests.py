#!/usr/bin/env python3
"""
Test runner script for RCPSP-CF-IVFTH.

This script provides a convenient way to run the test suite with different configurations.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def check_solver_availability():
    """Check which MILP solvers are available."""
    available_solvers = []
    
    try:
        from pyomo.environ import SolverFactory
        
        # Check CBC
        try:
            cbc = SolverFactory("cbc")
            if cbc.available():
                available_solvers.append("CBC")
        except:
            pass
        
        # Check GLPK
        try:
            glpk = SolverFactory("glpk")
            if glpk.available():
                available_solvers.append("GLPK")
        except:
            pass
        
        # Check Gurobi
        try:
            gurobi = SolverFactory("gurobi")
            if gurobi.available():
                available_solvers.append("Gurobi")
        except:
            pass
        
        # Check CPLEX
        try:
            cplex = SolverFactory("cplex")
            if cplex.available():
                available_solvers.append("CPLEX")
        except:
            pass
    
    except ImportError:
        print("WARNING: Pyomo not installed. Install with: pip install pyomo")
    
    return available_solvers


def run_tests(args):
    """Run the test suite with specified options."""
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add test directory
    cmd.append("tests/")
    
    # Add verbosity
    if args.verbose:
        cmd.append("-v")
    elif args.quiet:
        cmd.append("-q")
    
    # Add coverage if requested
    if args.coverage:
        cmd.extend(["--cov=rcpsp_cf_ivfth", "--cov-report=term-missing"])
        if args.html_coverage:
            cmd.append("--cov-report=html")
    
    # Add parallel execution if requested
    if args.parallel:
        cmd.extend(["-n", str(args.parallel)])
    
    # Filter tests by markers
    if args.fast:
        cmd.extend(["-m", "not slow"])
    
    if args.no_solver:
        cmd.extend(["-m", "not solver"])
    
    # Add specific test patterns
    if args.pattern:
        cmd.extend(["-k", args.pattern])
    
    # Add any additional pytest args
    if args.pytest_args:
        cmd.extend(args.pytest_args.split())
    
    print(f"Running command: {' '.join(cmd)}")
    print()
    
    # Check solver availability
    available_solvers = check_solver_availability()
    if available_solvers:
        print(f"Available MILP solvers: {', '.join(available_solvers)}")
    else:
        print("WARNING: No MILP solvers detected. Some tests may be skipped.")
        print("Install CBC: conda install -c conda-forge coincbc")
        print("Install GLPK: sudo apt-get install glpk-utils (Linux) or brew install glpk (macOS)")
    print()
    
    # Run the tests
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent)
        return result.returncode
    except KeyboardInterrupt:
        print("\nTest run interrupted by user.")
        return 1
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run RCPSP-CF-IVFTH test suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                    # Run all tests
  python run_tests.py --fast             # Skip slow tests
  python run_tests.py --coverage         # Run with coverage
  python run_tests.py --pattern="fuzzy"  # Run only fuzzy tests
  python run_tests.py --no-solver        # Skip solver-dependent tests
        """
    )
    
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "-q", "--quiet",
        action="store_true", 
        help="Quiet output"
    )
    
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Run with coverage analysis"
    )
    
    parser.add_argument(
        "--html-coverage",
        action="store_true",
        help="Generate HTML coverage report (requires --coverage)"
    )
    
    parser.add_argument(
        "-n", "--parallel",
        type=int,
        metavar="N",
        help="Run tests in parallel with N workers"
    )
    
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip slow tests"
    )
    
    parser.add_argument(
        "--no-solver",
        action="store_true",
        help="Skip tests that require MILP solvers"
    )
    
    parser.add_argument(
        "-k", "--pattern",
        metavar="PATTERN",
        help="Run only tests matching the given pattern"
    )
    
    parser.add_argument(
        "--pytest-args",
        metavar="ARGS",
        help="Additional arguments to pass to pytest"
    )
    
    args = parser.parse_args()
    
    if args.quiet and args.verbose:
        parser.error("Cannot specify both --quiet and --verbose")
    
    return run_tests(args)


if __name__ == "__main__":
    sys.exit(main())
