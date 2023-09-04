from typing import List
from websocket import create_connection, WebSocket
import json
import uuid

def request(request: str, data: dict = {}):
    """Format request"""
    if not bool(data):
        return json.dumps(
            {
                "id" : str(uuid.uuid4()),
                "request" : request
            }
        )
    else:
        return json.dumps(
            {
                "id" : str(uuid.uuid4()),
                "request" : request,
                "data" : data
            }
        )

def get_sessions(ws: WebSocket):
    """Get list of active sessions"""
    ws.send(request("get"))
    result = json.loads(ws.recv())

    if result["status"] == 0:
        return result["data"]["ids"]
    else:
        raise Exception(result["reason"])

def create_session(ws: WebSocket):
    """Create a new session"""
    ws.send(request("create/session"))
    result = json.loads(ws.recv())

    if result["status"] == 0:
        return result["data"]["id"], result["data"]["models"]
    else:
        raise Exception(result["reason"])

def remove_session(ws: WebSocket, session_id: str):
    """Remove an active session"""
    ws.send(request("remove/" + session_id))
    result = json.loads(ws.recv())

    if result["status"] != 0:
        raise Exception(result["reason"])

def connect_session(ws: WebSocket, session_id: str):
    """Connect to an active session"""
    ws.send(request("connect/" + session_id))
    result = json.loads(ws.recv())

    if result["status"] != 0:
        raise Exception(result["reason"])

def export_session_xml(ws: WebSocket, session_id: str):
    """Export session as XML"""
    ws.send(request("exportxml/" + session_id))
    result = json.loads(ws.recv())

    if result["status"] == 0:
        return result["data"]
    else:
        raise Exception(result["reason"])

def import_session_xml(ws: WebSocket, session_id: str, data: dict = {}):
    """Import session from XML"""
    ws.send(request("importxml/" + session_id, data))
    result = json.loads(ws.recv())

    if result["status"] != 0:
        raise Exception(result["reason"])

def save_session(ws: WebSocket, session_id: str):
    """Save session as binary data"""
    ws.send(request("save/" + session_id))
    result = json.loads(ws.recv())

    if result["status"] == 0:
        return result["data"]
    else:
        raise Exception(result["reason"])

def load_session(ws: WebSocket, data: dict = {}):
    """Load session from binary data"""
    ws.send(request("load/session", data))
    result = json.loads(ws.recv())

    if result["status"] == 0:
        return result["data"]["id"], result["data"]["models"]
    else:
        raise Exception(result["reason"])

def disconnect_session(ws: WebSocket, session_id: str):
    """Disconnect from an active session"""
    ws.send(request("disconnect/" + session_id))
    result = json.loads(ws.recv())

    if result["status"] != 0:
        raise Exception(result["reason"])

def create_BSplineCurve(ws: WebSocket, session_id: str,
                        degree: int = 1, init: int = 4, ncoeffs: List[int] = [4], nonuniform: bool = False):
    """Create BSpline curve"""
    data = {
        "degree"     : degree,
        "init"       : init,
        "ncoeffs"    : ncoeffs,
        "nonuniform" : nonuniform
    }
    ws.send(request("create/" + session_id + "/BSplineCurve", data))
    result = json.loads(ws.recv())

    if result["status"] == 0:
        return result["data"]["id"], result["data"]["model"]
    else:
        raise Exception(result["reason"])

def create_BSplineSurface(ws: WebSocket, session_id: str,
                          degree: int = 1, init: int = 4, ncoeffs: List[int] = [4, 4], nonuniform: bool = False):
    """Create BSpline surface"""
    data = {
        "degree"     : degree,
        "init"       : init,
        "ncoeffs"    : ncoeffs,
        "nonuniform" : nonuniform
    }
    ws.send(request("create/" + session_id + "/BSplineSurface", data))
    result = json.loads(ws.recv())

    if result["status"] == 0:
        return result["data"]["id"], result["data"]["model"]
    else:
        raise Exception(result["reason"])

def create_BSplineVolume(ws: WebSocket, session_id: str,
                         degree: int = 1, init: int = 4, ncoeffs: List[int] = [4, 4, 4], nonuniform: bool = False):
    """Create BSpline volume"""
    data = {
        "degree"     : degree,
        "init"       : init,
        "ncoeffs"    : ncoeffs,
        "nonuniform" : nonuniform
    }
    ws.send(request("create/" + session_id + "/BSplineVolume", data))
    result = json.loads(ws.recv())

    if result["status"] == 0:
        return result["data"]["id"], result["data"]["model"]
    else:
        raise Exception(result["reason"])

def remove_model(ws: WebSocket, session_id: str, instance: str):
    """Remove a model instance from an active session"""
    ws.send(request("remove/" + session_id + "/" + instance))
    result = json.loads(ws.recv())

    if result["status"] != 0:
        raise Exception(result["reason"])

def get_models(ws, session_id):
    """Get list of active models in session"""
    ws.send(request("get/" + session_id))
    result = json.loads(ws.recv())

    if result["status"] == 0:
        return result["data"]["ids"]
    else:
        raise Exception(result["reason"])

def get_model(ws: WebSocket, session_id: str, instance: str):
    """Get model data"""
    ws.send(request("get/" + session_id + "/" + instance))
    result = json.loads(ws.recv())

    if result["status"] == 0:
        return result["data"]
    else:
        raise Exception(result["reason"])

def get_model_component(ws: WebSocket, session_id: str, instance: str, component: str = ""):
    """Get model component data"""
    ws.send(request("get/" + session_id + "/" + instance + "/" + component))
    result = json.loads(ws.recv())

    if result["status"] == 0:
        return result["data"]
    else:
        raise Exception(result["reason"])

def get_model_attribute(ws: WebSocket, session_id: str, instance: str, component: str = "", attribute: str = ""):
    """Get model attribute data"""
    ws.send(request("get/" + session_id + "/" + instance + "/" + component + "/" + attribute))
    result = json.loads(ws.recv())

    if result["status"] == 0:
        return result["data"]
    else:
        raise Exception(result["reason"])

def put_model_attribute(ws: WebSocket, session_id: str, instance: str, component: str = "", attribute: str = "", data: dict = {}):
    """Put model attribute data"""
    ws.send(request("put/" + session_id + "/" + instance + "/" + component + "/" + attribute, data))
    result = json.loads(ws.recv())

    if result["status"] == 0:
        return result["data"]
    else:
        raise Exception(result["reason"])

def export_model_xml(ws: WebSocket, session_id: str, instance: str):
    """Export model as XML"""
    ws.send(request("exportxml/" + session_id + "/" + instance))
    result = json.loads(ws.recv())

    if result["status"] == 0:
        return result["data"]
    else:
        raise Exception(result["reason"])

def export_model_component_xml(ws: WebSocket, session_id: str, instance: str, component: str = ""):
    """Export model component as XML"""
    ws.send(request("exportxml/" + session_id + "/" + instance + "/" + component))
    result = json.loads(ws.recv())

    if result["status"] == 0:
        return result["data"]
    else:
        raise Exception(result["reason"])

def import_model_xml(ws: WebSocket, session_id: str, instance: str, data: dict = {}):
    """Import model from XML"""
    ws.send(request("importxml/" + session_id + "/" + instance, data))
    result = json.loads(ws.recv())

    if result["status"] != 0:
        raise Exception(result["reason"])

def import_model_component_xml(ws: WebSocket, session_id: str, instance: str, component: str = "", data: dict = {}):
    """Import model component from XML"""
    ws.send(request("importxml/" + session_id + "/" + instance + "/" + component, data))
    result = json.loads(ws.recv())

    if result["status"] != 0:
        raise Exception(result["reason"])

def save_model(ws: WebSocket, session_id: str, instance: str):
    """Save model as binary data"""
    ws.send(request("save/" + session_id + "/" + instance))
    result = json.loads(ws.recv())

    if result["status"] == 0:
        return result["data"]
    else:
        raise Exception(result["reason"])

def load_model(ws: WebSocket, session_id: str, data: dict = {}):
    """Load model from binary data"""
    ws.send(request("load/" + session_id, data))
    result = json.loads(ws.recv())

    if result["status"] == 0:
        return result["data"]["id"], result["data"]["model"]
    else:
        raise Exception(result["reason"])

def main():
    """Main function"""

    # Establish connection
    ws = create_connection("ws://localhost:9001")

    for session_id in get_sessions(ws):
        print("Session id: {}".format(session_id))

        for instance in get_models(ws, session_id):
            model = get_model(ws, session_id, instance)

            print("  {} {}".format(instance, model["model"]["description"]))

    ws.close()

if __name__ == "__main__":
    main()
