"""
Tests for the examples module and toy instance functionality.
"""

import pytest
from rcpsp_cf_ivfth.examples import build_toy_instance
from rcpsp_cf_ivfth import Activity, FinanceParams, CalendarParams, NIVTF


class TestToyInstance:
    """Test the toy instance builder."""
    
    def test_build_toy_instance_structure(self):
        """Test that build_toy_instance returns correct structure."""
        activities, finance, calendar = build_toy_instance()
        
        # Check types
        assert isinstance(activities, dict)
        assert isinstance(finance, FinanceParams)
        assert isinstance(calendar, CalendarParams)
        
        # Check activities
        assert len(activities) == 4  # Start, A1, A2, End
        assert "Start" in activities
        assert "A1" in activities
        assert "A2" in activities
        assert "End" in activities
        
        # Check that all activities are Activity objects
        for name, activity in activities.items():
            assert isinstance(activity, Activity)
            assert activity.name == name
    
    def test_toy_instance_precedence_relationships(self):
        """Test that precedence relationships are set correctly."""
        activities, _, _ = build_toy_instance()
        
        # Check precedence structure
        assert activities["Start"].predecessors == []
        assert activities["A1"].predecessors == ["Start"]
        assert activities["A2"].predecessors == ["A1"]
        assert activities["End"].predecessors == ["A2"]
    
    def test_toy_instance_modes(self):
        """Test that activities have correct modes."""
        activities, _, _ = build_toy_instance()
        
        # Start and End should have single mode
        assert len(activities["Start"].modes) == 1
        assert len(activities["End"].modes) == 1
        assert 1 in activities["Start"].modes
        assert 1 in activities["End"].modes
        
        # A1 and A2 should have two modes each
        assert len(activities["A1"].modes) == 2
        assert len(activities["A2"].modes) == 2
        assert 1 in activities["A1"].modes
        assert 2 in activities["A1"].modes
        assert 1 in activities["A2"].modes
        assert 2 in activities["A2"].modes
    
    def test_toy_instance_resource_structure(self):
        """Test that resources are structured correctly."""
        activities, finance, _ = build_toy_instance()
        
        # Check finance has correct resource costs
        assert len(finance.CR_k) == 2  # Renewable resources 1, 2
        assert len(finance.CW_l) == 1  # Non-renewable resource 1
        assert 1 in finance.CR_k
        assert 2 in finance.CR_k
        assert 1 in finance.CW_l
        
        # Check that activities reference these resources
        for activity in activities.values():
            for mode in activity.modes.values():
                # Check renewable resources
                for k in mode.renewables.keys():
                    assert k in finance.CR_k, f"Resource {k} not in cost dictionary"
                
                # Check non-renewable resources
                for l in mode.nonrenewables.keys():
                    assert l in finance.CW_l, f"Resource {l} not in cost dictionary"
    
    def test_toy_instance_fuzzy_numbers(self):
        """Test that all NIVTF numbers are valid."""
        activities, _, _ = build_toy_instance()
        
        for activity_name, activity in activities.items():
            for mode_id, mode in activity.modes.items():
                # Check duration is valid NIVTF
                assert isinstance(mode.duration, NIVTF)
                
                # Check renewable resource NIVTFs
                for k, nivtf in mode.renewables.items():
                    assert isinstance(nivtf, NIVTF)
                    # Should not raise validation errors
                    assert nivtf.E1_L() >= 0
                    assert nivtf.E2_L() >= 0
                
                # Check non-renewable resource NIVTFs
                for l, nivtf in mode.nonrenewables.items():
                    assert isinstance(nivtf, NIVTF)
                    assert nivtf.E1_L() >= 0
                    assert nivtf.E2_L() >= 0
    
    def test_toy_instance_payments(self):
        """Test that payment amounts are reasonable."""
        activities, _, _ = build_toy_instance()
        
        # Start and End should have zero payments
        assert activities["Start"].modes[1].payment == 0.0
        assert activities["End"].modes[1].payment == 0.0
        
        # A1 and A2 should have positive payments
        for mode in activities["A1"].modes.values():
            assert mode.payment > 0
        
        for mode in activities["A2"].modes.values():
            assert mode.payment > 0
        
        # Mode 2 should generally have higher payment than mode 1 (faster/more expensive)
        assert activities["A1"].modes[2].payment > activities["A1"].modes[1].payment
        assert activities["A2"].modes[2].payment > activities["A2"].modes[1].payment
    
    def test_toy_instance_calendar_consistency(self):
        """Test that calendar parameters are consistent."""
        _, _, calendar = build_toy_instance()
        
        assert calendar.T_days == 60
        assert len(calendar.Y_periods) == 2
        
        # Periods should cover the entire time horizon
        assert calendar.Y_periods[0] == (1, 30)
        assert calendar.Y_periods[1] == (31, 60)
        
        # Should cover exactly T_days
        total_days = sum(end - start + 1 for start, end in calendar.Y_periods)
        assert total_days == calendar.T_days
    
    def test_toy_instance_finance_parameters(self):
        """Test that finance parameters are reasonable."""
        _, finance, _ = build_toy_instance()
        
        # Interest rates should be positive but reasonable
        assert 0 <= finance.alpha_excess_cash <= 0.1  # Max 10% per period
        assert 0 <= finance.beta_delayed_pay <= 0.2   # Max 20% per period
        assert 0 <= finance.gamma_LTL <= 0.1
        assert 0 <= finance.delta_STL <= 0.1
        
        # Capital and limits should be positive
        assert finance.IC > 0
        assert finance.max_LTL > 0
        assert finance.max_STL > 0
        assert finance.CC_daily_cap > 0
        
        # Resource costs should be positive
        for cost in finance.CR_k.values():
            assert cost > 0
        for cost in finance.CW_l.values():
            assert cost > 0


class TestExampleIntegration:
    """Test integration between example and main classes."""
    
    def test_toy_instance_with_solver_class(self, solver_name):
        """Test that toy instance works with the main solver class."""
        from rcpsp_cf_ivfth import RCPSP_CF_IVFTH, IVFTHTargets, IVFTHWeights
        
        activities, finance, calendar = build_toy_instance()
        
        # Should be able to create solver instance
        ivfth = RCPSP_CF_IVFTH(activities, finance, calendar)
        
        # Should be able to build model
        targets = IVFTHTargets(
            alpha_level=0.5,
            Z1_PIS=10.0, Z1_NIS=60.0,
            Z2_PIS=30000.0, Z2_NIS=0.0
        )
        weights = IVFTHWeights(theta1=0.5, theta2=0.5, gamma_tradeoff=0.5)
        
        model = ivfth.build_model(targets, weights)
        
        # Should be able to solve (if solver available)
        result = ivfth.solve(model, solver_name=solver_name)
        
        # Should get a valid result
        assert "status" in result
        assert "objective" in result
        assert "Cmax" in result
        assert "CF_final" in result


class TestExampleVariations:
    """Test variations of the toy instance for robustness."""
    
    def test_modified_toy_instance(self):
        """Test that we can modify the toy instance and still use it."""
        activities, finance, calendar = build_toy_instance()
        
        # Modify some parameters
        finance.IC = 15000.0  # More initial capital
        finance.alpha_excess_cash = 0.02  # Higher interest
        
        # Add more time
        calendar.T_days = 90
        calendar.Y_periods = [(1, 30), (31, 60), (61, 90)]
        
        # Should still work
        from rcpsp_cf_ivfth import RCPSP_CF_IVFTH, IVFTHTargets, IVFTHWeights
        
        ivfth = RCPSP_CF_IVFTH(activities, finance, calendar)
        
        targets = IVFTHTargets(
            alpha_level=0.3,
            Z1_PIS=15.0, Z1_NIS=90.0,
            Z2_PIS=40000.0, Z2_NIS=0.0
        )
        weights = IVFTHWeights(theta1=0.7, theta2=0.3, gamma_tradeoff=0.8)
        
        # Should build without errors
        model = ivfth.build_model(targets, weights)
        assert model is not None
    
    def test_minimal_instance(self):
        """Test with a minimal instance (just Start -> End)."""
        from rcpsp_cf_ivfth import Activity, ModeData, FinanceParams, CalendarParams
        from rcpsp_cf_ivfth.fuzzy import NIVTF, create_triangle
        
        # Create minimal instance
        zero_nivtf = NIVTF(*create_triangle(0, 0, 0))
        
        activities = {
            "Start": Activity(
                name="Start", predecessors=[],
                modes={1: ModeData(
                    duration=zero_nivtf,
                    renewables={1: zero_nivtf},
                    nonrenewables={1: zero_nivtf},
                    payment=0.0
                )}
            ),
            "End": Activity(
                name="End", predecessors=["Start"],
                modes={1: ModeData(
                    duration=zero_nivtf,
                    renewables={1: zero_nivtf},
                    nonrenewables={1: zero_nivtf},
                    payment=0.0
                )}
            )
        }
        
        finance = FinanceParams(
            alpha_excess_cash=0.01, beta_delayed_pay=0.05,
            gamma_LTL=0.06, delta_STL=0.08,
            IC=1000.0, max_LTL=500.0, max_STL=300.0, min_CF=0.0,
            CC_daily_cap=100.0,
            CR_k={1: 1.0}, CW_l={1: 1.0}
        )
        
        calendar = CalendarParams(T_days=10, Y_periods=[(1, 10)])
        
        # Should work with minimal instance
        from rcpsp_cf_ivfth import RCPSP_CF_IVFTH
        ivfth = RCPSP_CF_IVFTH(activities, finance, calendar)
        
        # Should be able to validate
        assert len(ivfth.activities) == 2
        assert ivfth.finance.IC == 1000.0
        assert ivfth.calendar.T_days == 10
