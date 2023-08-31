import os
import unittest
from wsiganet import *
import xml.etree.ElementTree as ET

class TestSession(unittest.TestCase):

    def setUp(self):

        # Establish connection
        self.ws = create_connection("ws://localhost:9001")

        # Get list of sessions
        self.session_ids = get_sessions(self.ws)

        # Create a new session
        self.session_id, _ = create_session(self.ws)

    def tearDown(self):

        # Remove session
        remove_session(self.ws, self.session_id)
        
        # Check that list of sessions has not changed
        self.assertEqual(self.session_ids, get_sessions(self.ws))

        # Close connection
        self.ws.close()

    def test_create_remove(self):
    
        # Check that list of sessions has changed
        self.assertNotEqual(self.session_ids, get_sessions(self.ws))        

    def test_connect_disconnect(self):

        # Check that list of sessions has changed
        self.assertNotEqual(self.session_ids, get_sessions(self.ws))

        # Disconnect from session
        disconnect_session(self.ws, self.session_id)

        # Check that list of sessions has changed
        self.assertNotEqual(self.session_ids, get_sessions(self.ws))

        # Reconnect to session
        connect_session(self.ws, self.session_id)

    def test_broadcast(self):

        # Establish connection
        ws = create_connection("ws://localhost:9001")

        # Connect to existing session
        connect_session(ws, self.session_id)

        # Create a new BSpline surface
        model, _ = create_BSplineSurface(self.ws, self.session_id)

        # Receive broadcast message
        self.assertTrue(ws.recv())

        # Disconnect from session
        disconnect_session(self.ws, self.session_id)
        
        # Close connection
        ws.close()

    def test_exportxml(self):

        # Create three new BSpline objects
        model0, _ = create_BSplineSurface(self.ws, self.session_id)
        model1, _ = create_BSplineCurve(self.ws, self.session_id)
        model2, _ = create_BSplineVolume(self.ws, self.session_id)
        
        data = export_session_xml(self.ws, self.session_id)        
        xml1 = ET.fromstring(data["xml"])

        xml2 = ET.parse(os.path.join(os.path.dirname(__file__), "Session.xml"))

        self.assertEqual(ET.tostring(xml1), ET.tostring(xml2.getroot()))    

        # Remove model instances
        remove_model(self.ws, self.session_id, model0)
        remove_model(self.ws, self.session_id, model1)
        remove_model(self.ws, self.session_id, model2)
        
class TestBSplineSurface(unittest.TestCase):

    def setUp(self):

        # Establish connection
        self.ws = create_connection("ws://localhost:9001")

        # Get list of sessions
        self.session_ids = get_sessions(self.ws)

        # Create a new session
        self.session_id, _ = create_session(self.ws)

        # Get list of model instances
        self.models = get_models(self.ws, self.session_id)

        # Create a new BSpline surface
        self.model, self.data = create_BSplineSurface(self.ws, self.session_id, degree = 1, init = 4, ncoeffs = [3, 2])

        # Get component name
        self.component = self.data["inputs"][0]["name"]

    def tearDown(self):

        # Remove model instances
        remove_model(self.ws, self.session_id, self.model)
        
        # Check that list of model instances has not changed
        self.assertEqual(self.models, get_models(self.ws, self.session_id))

        # Remove session
        remove_session(self.ws, self.session_id)
        
        # Check that list of sessions has not changed
        self.assertEqual(self.session_ids, get_sessions(self.ws))

        # Close connection
        self.ws.close()
    
    def test_create_remove(self):

        # Check that list of model instances has changed
        self.assertNotEqual(self.models, get_models(self.ws, self.session_id))

    def test_get_component(self):

        # Get component
        component = get_model_component(self.ws, self.session_id, self.model, self.component)

        self.assertTrue(component)

    def test_get_attributes(self):

        # Check degrees
        self.assertEqual(get_model_attribute(self.ws, self.session_id, self.model, self.component, "degrees")["degrees"],
                         [1, 1])

        # Check number of coefficients
        self.assertEqual(get_model_attribute(self.ws, self.session_id, self.model, self.component, "ncoeffs")["ncoeffs"],
                         [3, 2])

        # Check number of knots
        self.assertEqual(get_model_attribute(self.ws, self.session_id, self.model, self.component, "nknots")["nknots"],
                         [5, 4])

        # Check coefficients
        self.assertEqual(get_model_attribute(self.ws, self.session_id, self.model, self.component, "coeffs")["coeffs"],
                         [[0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
                          [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                          [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])

        # Check knots
        self.assertEqual(get_model_attribute(self.ws, self.session_id, self.model, self.component, "knots")["knots"],
                         [[0.0, 0.0, 0.5, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]])

    def test_put_attributes(self):
        
        # Check coefficients
        self.assertEqual(get_model_attribute(self.ws, self.session_id, self.model, self.component, "coeffs")["coeffs"],
                         [[0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
                          [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                          [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])

        # Change coefficients
        put_model_attribute(self.ws, self.session_id, self.model, self.component, "coeffs", { "indices" : [0, 3],
                                                                                              "coeffs"  : [[0.0, 0.0, 0.0], [0.5, 1.0, 0.5]]} )

        # Check updated coefficients
        self.assertEqual(get_model_attribute(self.ws, self.session_id, self.model, self.component, "coeffs")["coeffs"],
                         [[0.0, 0.5, 1.0, 0.5, 0.5, 1.0],
                          [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                          [0.0, 1.0, 1.0, 0.5, 1.0, 1.0]])

    def test_exportxml(self):

        # Export model to XML
        data = export_model_xml(self.ws, self.session_id, self.model)
        xml1 = ET.fromstring(data["xml"])
        xml2 = ET.parse(os.path.join(os.path.dirname(__file__), "BSplineSurface.xml"))

        self.assertEqual(ET.tostring(xml1), ET.tostring(xml2.getroot()))

    def test_exportxml_component(self):

        # Export model component to XML
        data = export_model_component_xml(self.ws, self.session_id, self.model, "geometry")
        xml1 = ET.fromstring(data["xml"])        
        xml2 = ET.parse(os.path.join(os.path.dirname(__file__), "BSplineSurfaceComponent.xml"))

        self.assertEqual(ET.tostring(xml1), ET.tostring(xml2.getroot()))

    def test_importxml(self):

        # Change coefficients
        put_model_attribute(self.ws, self.session_id, self.model, self.component, "coeffs", { "indices" : [0, 3],
                                                                                              "coeffs"  : [[0.0, 0.0, 0.0], [0.5, 1.0, 0.5]]} )

        # Export model to XML
        data = export_model_xml(self.ws, self.session_id, self.model)
        xml1 = ET.fromstring(data["xml"])
        xml2 = ET.parse(os.path.join(os.path.dirname(__file__), "BSplineSurface.xml"))

        self.assertNotEqual(ET.tostring(xml1), ET.tostring(xml2.getroot()))

        # Import model from XML        
        result = import_model_xml(self.ws, self.session_id, self.model, { "xml" : str(ET.tostring(xml2.getroot())) })

        # Export model to XML
        data = export_model_xml(self.ws, self.session_id, self.model)
        xml1 = ET.fromstring(data["xml"])

        self.assertEqual(ET.tostring(xml1), ET.tostring(xml2.getroot()))

    def test_importxml_component(self):

        # Change coefficients
        put_model_attribute(self.ws, self.session_id, self.model, self.component, "coeffs", { "indices" : [0, 3],
                                                                                              "coeffs"  : [[0.0, 0.0, 0.0], [0.5, 1.0, 0.5]]} )

        # Export model to XML
        data = export_model_xml(self.ws, self.session_id, self.model)
        xml1 = ET.fromstring(data["xml"])
        xml2 = ET.parse(os.path.join(os.path.dirname(__file__), "BSplineSurfaceComponent.xml"))

        self.assertNotEqual(ET.tostring(xml1), ET.tostring(xml2.getroot()))

        # Import model from XML        
        result = import_model_component_xml(self.ws, self.session_id, self.model, "geometry", { "xml" : str(ET.tostring(xml2.getroot())) })

        # Export model to XML
        data = export_model_component_xml(self.ws, self.session_id, self.model, "geometry")
        xml1 = ET.fromstring(data["xml"])

        self.assertEqual(ET.tostring(xml1), ET.tostring(xml2.getroot()))
        
if __name__ == '__main__':
    unittest.main()
