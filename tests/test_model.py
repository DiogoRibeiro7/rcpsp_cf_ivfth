"""
Tests for the MILP model building and validation.
"""

import pytest
from pyomo.environ import value, ConcreteModel
from pyomo.opt import TerminationCondition

from rcpsp_cf_ivfth import RCPSP_CF_IVFTH


class TestModelValidation:
    """Test model input validation."""
    
    def test_missing_start_end_activities(self, toy_instance):
        """Test that missing Start/End activities are caught."""
        activities, finance, calendar = toy_instance
        
        # Remove Start activity
        activities_no_start = {k: v for k, v in activities.items() if k != "Start"}
        
        with pytest.raises(ValueError, match="Activities must include 'Start' and 'End'"):
            RCPSP_CF_IVFTH(activities_no_start, finance, calendar)
        
        # Remove End activity
        activities_no_end = {k: v for k, v in activities.items() if k != "End"}
        
        with pytest.raises(ValueError, match="Activities must include 'Start' and 'End'"):
            RCPSP_CF_IVFTH(activities_no_end, finance, calendar)
    
    def test_unknown_predecessor(self, toy_instance):
        """Test that unknown predecessors are caught."""
        activities, finance, calendar = toy_instance
        
        # Add an activity with unknown predecessor
        from rcpsp_cf_ivfth import Activity, ModeData, NIVTF, create_triangle
        
        bad_activity = Activity(
            name="BadActivity", 
            predecessors=["NonExistentActivity"],
            modes={
                1: ModeData(
                    duration=NIVTF(*create_triangle(5, 7, 9)),
                    renewables={1: NIVTF(*create_triangle(1, 2, 3))},
                    nonrenewables={1: NIVTF(*create_triangle(0, 1, 2))},
                    payment=1000.0
                )
            }
        )
        
        activities["BadActivity"] = bad_activity
        
        with pytest.raises(ValueError, match="Unknown predecessor 'NonExistentActivity'"):
            RCPSP_CF_IVFTH(activities, finance, calendar)
    
    def test_calendar_period_validation(self, toy_instance):
        """Test that calendar period validation works."""
        activities, finance, calendar = toy_instance
        
        # Calendar with T_days too small for Y_periods
        from rcpsp_cf_ivfth import CalendarParams
        
        bad_calendar = CalendarParams(
            T_days=30,  # Too small
            Y_periods=[(1, 30), (31, 60)]  # Goes up to day 60
        )
        
        with pytest.raises(ValueError, match="T_days must cover all Y_periods upper bounds"):
            RCPSP_CF_IVFTH(activities, finance, bad_calendar)
    
    def test_empty_periods_validation(self, toy_instance):
        """Test that empty Y_periods list is caught."""
        activities, finance, calendar = toy_instance
        
        from rcpsp_cf_ivfth import CalendarParams
        
        bad_calendar = CalendarParams(
            T_days=60,
            Y_periods=[]  # Empty
        )
        
        with pytest.raises(ValueError, match="At least one long period \\(Y_periods\\) is required"):
            RCPSP_CF_IVFTH(activities, finance, bad_calendar)


class TestModelBuilding:
    """Test that the model can be built without errors."""
    
    def test_model_builds_successfully(self, toy_instance, ivfth_targets, ivfth_weights):
        """Test that the model builds without errors."""
        activities, finance, calendar = toy_instance
        
        ivfth = RCPSP_CF_IVFTH(activities, finance, calendar)
        model = ivfth.build_model(ivfth_targets, ivfth_weights)
        
        # Should create a Pyomo ConcreteModel
        assert isinstance(model, ConcreteModel)
        
        # Check that key variables exist
        assert hasattr(model, 'X')  # Start variables
        assert hasattr(model, 'Xp')  # Completion variables
        assert hasattr(model, 'CF')  # Cash flow variables
        assert hasattr(model, 'Cmax')  # Makespan variable
        assert hasattr(model, 'mu1')  # Membership variables
        assert hasattr(model, 'mu2')
        assert hasattr(model, 'lambda_star')
        
        # Check that key constraints exist
        assert hasattr(model, 'start_once')  # Each activity starts once
        assert hasattr(model, 'precedence')  # Precedence constraints
        assert hasattr(model, 'cf1')  # First period cash flow
        assert hasattr(model, 'OBJ')  # Objective function
    
    def test_model_sets_and_params(self, toy_instance, ivfth_targets, ivfth_weights):
        """Test that model sets and parameters are created correctly."""
        activities, finance, calendar = toy_instance
        
        ivfth = RCPSP_CF_IVFTH(activities, finance, calendar)
        model = ivfth.build_model(ivfth_targets, ivfth_weights)
        
        # Check sets
        assert len(model.I) == 4  # Start, A1, A2, End
        assert len(model.T) == 60  # 60 days
        assert len(model.Y) == 2   # 2 periods
        assert len(model.K) == 2   # 2 renewable resources
        assert len(model.L) == 1   # 1 non-renewable resource
        
        # Check key parameters
        assert value(model.IC) == 10000.0  # Initial capital
        assert value(model.CC) == 2000.0   # Daily cost cap


class TestModelConstraints:
    """Test specific model constraints and their logic."""
    
    def test_alpha_level_propagation(self, toy_instance, ivfth_weights):
        """Test that alpha level is correctly propagated to the model."""
        activities, finance, calendar = toy_instance
        
        from rcpsp_cf_ivfth import IVFTHTargets
        
        # Test with different alpha levels
        for alpha in [0.0, 0.3, 0.7, 1.0]:
            targets = IVFTHTargets(
                alpha_level=alpha,
                Z1_PIS=10.0, Z1_NIS=60.0,
                Z2_PIS=30000.0, Z2_NIS=0.0
            )
            
            ivfth = RCPSP_CF_IVFTH(activities, finance, calendar)
            model = ivfth.build_model(targets, ivfth_weights)
            
            # Check that alpha parameter is set correctly
            assert value(model.alpha) == pytest.approx(alpha)
    
    def test_membership_bounds(self, toy_instance, ivfth_targets, ivfth_weights):
        """Test that membership variables have correct bounds."""
        activities, finance, calendar = toy_instance
        
        ivfth = RCPSP_CF_IVFTH(activities, finance, calendar)
        model = ivfth.build_model(ivfth_targets, ivfth_weights)
        
        # Membership variables should be bounded in [0,1]
        assert model.mu1.bounds == (0.0, 1.0)
        assert model.mu2.bounds == (0.0, 1.0)
        assert model.lambda_star.bounds == (0.0, 1.0)
    
    def test_activity_mode_mapping(self, toy_instance, ivfth_targets, ivfth_weights):
        """Test that activity modes are mapped correctly."""
        activities, finance, calendar = toy_instance
        
        ivfth = RCPSP_CF_IVFTH(activities, finance, calendar)
        model = ivfth.build_model(ivfth_targets, ivfth_weights)
        
        # Check that activities have correct number of modes
        assert len(model.M_i["Start"]) == 1
        assert len(model.M_i["A1"]) == 2
        assert len(model.M_i["A2"]) == 2
        assert len(model.M_i["End"]) == 1
        
        # Check that payment parameters are set correctly
        # Start and End should have 0 payment
        assert value(model.PA_im[("Start", 1)]) == 0.0
        assert value(model.PA_im[("End", 1)]) == 0.0
        
        # A1 modes should have their respective payments
        assert value(model.PA_im[("A1", 1)]) == 3000.0
        assert value(model.PA_im[("A1", 2)]) == 3600.0


class TestHelperMethods:
    """Test the helper methods in RCPSP_CF_IVFTH class."""
    
    def test_duration_alpha_l(self):
        """Test the _duration_alpha_L helper method."""
        from rcpsp_cf_ivfth.fuzzy import NIVTF, create_triangle
        
        # Create a test NIVTF
        nivtf = NIVTF(*create_triangle(5, 7, 9, widen=0.5))
        
        # Test with alpha = 0.5
        E1L, E2L = RCPSP_CF_IVFTH._duration_alpha_L(nivtf, alpha=0.5, use_lower=True)
        
        # Should return E1_L and E2_L
        assert E1L == pytest.approx(nivtf.E1_L())
        assert E2L == pytest.approx(nivtf.E2_L())
        
        # Test alpha-blended duration: alpha*E2_L + (1-alpha)*E1_L
        alpha = 0.3
        E1L, E2L = RCPSP_CF_IVFTH._duration_alpha_L(nivtf, alpha=alpha, use_lower=True)
        expected_blend = alpha * E2L + (1.0 - alpha) * E1L
        
        # This should match the computation in precedence constraints
        assert expected_blend == pytest.approx(alpha * nivtf.E2_L() + (1.0 - alpha) * nivtf.E1_L())
    
    def test_res_use_alpha_l(self):
        """Test the _res_use_alpha_L helper method."""
        from rcpsp_cf_ivfth.fuzzy import NIVTF, create_triangle
        
        nivtf = NIVTF(*create_triangle(2, 4, 6, widen=0.4))
        
        # Test resource usage blending: (1-alpha)*E2_L + alpha*E1_L
        alpha = 0.6
        result = RCPSP_CF_IVFTH._res_use_alpha_L(nivtf, alpha=alpha)
        
        expected = (1.0 - alpha) * nivtf.E2_L() + alpha * nivtf.E1_L()
        assert result == pytest.approx(expected)
    
    def test_dur_mid_ev(self):
        """Test the _dur_mid_EV helper method."""
        from rcpsp_cf_ivfth.fuzzy import NIVTF, create_triangle
        
        nivtf = NIVTF(*create_triangle(3, 5, 7, widen=0.3))
        
        result = RCPSP_CF_IVFTH._dur_mid_EV(nivtf)
        expected = nivtf.EV_mid()
        
        assert result == pytest.approx(expected)
