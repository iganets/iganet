import unittest
from wsiganet import *

class TestSession(unittest.TestCase):

    def test_create_remove(self):

        # Establish connection
        ws = create_connection("ws://localhost:9001")

        # Get list of sessions
        session_ids = get_sessions(ws)

        # Create a new session
        session_id, _ = create_session(ws)

        # Check that list of sessions has changed
        self.assertNotEqual(session_ids, get_sessions(ws))

        # Remove session
        remove_session(ws, session_id)

        # Check that list of sessions has not changed
        self.assertEqual(session_ids, get_sessions(ws))

        # Close connection
        ws.close()

    def test_connect_disconnect(self):

        # Establish connection
        ws = create_connection("ws://localhost:9001")

        # Get list of sessions
        session_ids = get_sessions(ws)

        # Create a new session
        session_id, _ = create_session(ws)

        # Check that list of sessions has changed
        self.assertNotEqual(session_ids, get_sessions(ws))

        # Disconnect from session
        disconnect_session(ws, session_id)

        # Check that list of sessions has changed
        self.assertNotEqual(session_ids, get_sessions(ws))

        # Reconnect to session
        connect_session(ws, session_id)

        # Check that list of sessions has changed
        self.assertNotEqual(session_ids, get_sessions(ws))

        # Remove session
        remove_session(ws, session_id)

        # Check that list of sessions has not changed
        self.assertEqual(session_ids, get_sessions(ws))

        # Close connection
        ws.close()

    def test_broadcast(self):
        import websocket

        # Establish connections
        ws1 = create_connection("ws://localhost:9001")
        ws2 = create_connection("ws://localhost:9001")

        # Create a new session
        session_id, _ = create_session(ws1)

        # Connect to existing session
        connect_session(ws2, session_id)

        # Create a new BSpline surface
        model, _ = create_BSplineSurface(ws2, session_id)

        # Receive broadcast message
        self.assertTrue(ws1.recv())

        # Close connections
        ws1.close()
        ws2.close()

class TestBSplineSurface(unittest.TestCase):

    def test_create_remove(self):

        # Establish connection
        ws = create_connection("ws://localhost:9001")

        # Create a new session
        session_id, _ = create_session(ws)

        # Get list of model models
        models = get_models(ws, session_id)

        # Create a new BSpline surface
        model, _ = create_BSplineSurface(ws, session_id, degree = 2, init = 4, ncoeffs = [10, 5])

        # Check that list of model models has changed
        self.assertNotEqual(models, get_models(ws, session_id))

        # Remove model model
        remove_model(ws, session_id, model)

        # Check that list of model models has not changed
        self.assertEqual(models, get_models(ws, session_id))

        # Remove session
        remove_session(ws, session_id)

        # Close connection
        ws.close()

    def test_get_component(self):

        # Establish connection
        ws = create_connection("ws://localhost:9001")

        # Create a new session
        session_id, _ = create_session(ws)

        # Create a new BSpline surface
        model, data = create_BSplineSurface(ws, session_id, degree = 2, init = 4, ncoeffs = [10, 5])

        # Get component name
        component = data["inputs"][0]["name"]

        # Get component
        component = get_model_component(ws, session_id, model, component)

        self.assertTrue(component)

         # Remove model model
        remove_model(ws, session_id, model)

        # Remove session
        remove_session(ws, session_id)

        # Close connection
        ws.close()

    def test_get_attributes(self):

        # Establish connection
        ws = create_connection("ws://localhost:9001")

        # Create a new session
        session_id, _ = create_session(ws)

        # Create a new BSpline surface
        model, data = create_BSplineSurface(ws, session_id, degree = 1, init = 4, ncoeffs = [3, 2])

        # Get component name
        component = data["inputs"][0]["name"]

        # Check degrees
        self.assertEqual(get_model_attribute(ws, session_id, model, component, "degrees")["degrees"],
                         [1, 1])

        # Check number of coefficients
        self.assertEqual(get_model_attribute(ws, session_id, model, component, "ncoeffs")["ncoeffs"],
                         [3, 2])

        # Check number of knots
        self.assertEqual(get_model_attribute(ws, session_id, model, component, "nknots")["nknots"],
                         [5, 4])

        # Check coefficients
        self.assertEqual(get_model_attribute(ws, session_id, model, component, "coeffs")["coeffs"],
                         [[0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
                          [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                          [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])

        # Check knots
        self.assertEqual(get_model_attribute(ws, session_id, model, component, "knots")["knots"],
                         [[0.0, 0.0, 0.5, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]])

         # Remove model model
        remove_model(ws, session_id, model)

        # Remove session
        remove_session(ws, session_id)

        # Close connection
        ws.close()

    def test_put_attributes(self):

        # Establish connection
        ws = create_connection("ws://localhost:9001")

        # Create a new session
        session_id, _ = create_session(ws)

        # Create a new BSpline surface
        model, data = create_BSplineSurface(ws, session_id, degree = 1, init = 4, ncoeffs = [3, 2])

        # Get component name
        component = data["inputs"][0]["name"]

        # Check coefficients
        self.assertEqual(get_model_attribute(ws, session_id, model, component, "coeffs")["coeffs"],
                         [[0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
                          [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                          [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])

        # Change coefficients
        put_model_attribute(ws, session_id, model, component, "coeffs", { "indices" : [0, 3],
                                                                          "coeffs"  : [[0.0, 0.0, 0.0], [0.5, 1.0, 0.5]]} )

        # Check updated coefficients
        self.assertEqual(get_model_attribute(ws, session_id, model, component, "coeffs")["coeffs"],
                         [[0.0, 0.5, 1.0, 0.5, 0.5, 1.0],
                          [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                          [0.0, 1.0, 1.0, 0.5, 1.0, 1.0]])

        # Remove model model
        remove_model(ws, session_id, model)

        # Remove session
        remove_session(ws, session_id)

        # Close connection
        ws.close()

if __name__ == '__main__':
    unittest.main()
