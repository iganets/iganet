# WebApp protocol

All WebApps implement the following [WebSocket](https://en.wikipedia.org/wiki/WebSocket)-based client-server protocol. All communication is initiated by the client and responded by the server. Only broadcasts are initiated by the server and responded by all clients, e.g., when a client request leads to a state change of the server, then an update is broadcasted to all clients connected to the same session.

The format of the [JSON](https://en.wikipedia.org/wiki/JSON)-based protocol is as follows:

_Client request_

```
{
    "id"      : UUID
    "request" : command[/session-id][/object-id][/subcommand]
  [ "data"    : {...} ]
}
```

_Server response_

```
{
    "request" : UUID
    "status"  : <integer>
  [ "reason"  : <string> ]
  [ "data"    : {...} ]
}
```

*   `UUID` is chosen by the client for each new request and copied by the server in the reply. 

*   `request` encodes the actual `command`, optionally followed by the `session-id`, the `object-id`, and a `subcommand`

*   `status` is an integer from the following list of predefined integers:
    - 0 : `SUCCESS`
    - 1 : `ERROR`

*   `reason` is an optional string that contains human-readable information about a failed request. Successful requests with `status : 0` will not contain a `reason` field

*    `data` is an optional request/response-dependent payload

In what follows, only the non-generic parts of the protocol are specified in more detail. 

## Creation command - `create/`

1) Creation of a new __session__

   _Client request_
   ```
     "request" : create/session
   ```

   _Server response_
   ```
     "status"  : 0
     "data"    : { "id" : session-id }
   ```
   or
   ```
     "status"  : 1
     "reason"  : <string>
   ```

2) Creation of a new __object__

   _Client request_
   ```
   "request" : create/session-id/<object-type>
   "data"    : {...}
   ```

   Supported __object-types__:

   -   `uniformBSpline`
       ```
       "data" : { 
                  "geoDim"  : <integer> valid values are 1,2,3,4
                  "degrees" : [<comma-separated list of integers>]
                  "ncoeffs" : [<comma-separated list of integers>]
                  "init"    : <integer> valid values are
                                         0 : initialize coefficients by zeros,
                                         1 : initialize coefficients by ones,
                                         2 : initialize coefficients by 0,1,...,ncoeff-1,
                                         3 : initialize coefficients by random numbers,
                                         4 : initialize coefficients by Greville abscissae
                }
       ```

   -   `nonuniformBSpline`
        ```
       "data" : { 
                  "geoDim"  : <integer> valid values are 1,2,3,4
                  "degrees" : [<comma-separated list of integers>]
                  "knots"   : [[<comma-separated list of lists of integers>]]
                  "init"    : <integer> valid values are
                                         0 : initialize coefficients by zeros,
                                         1 : initialize coefficients by ones,
                                         2 : initialize coefficients by 0,1,...,ncoeff-1,
                                         3 : initialize coefficients by random numbers,
                                         4 : initialize coefficients by Greville abscissae
                }
       ```

   -   More object types will be added

   _Server response_
   ```
   "status"  : 0
   "data"    : { "id" : object-id }
   ```
   or
   ```
     "status"  : 1
     "reason"  : <string>
   ```

## Removal commands - `remove/`

1) Removal of an existing __session__

   _Client request_
   ```
   "request" : remove/session-id
   ```

   _Server response_
   ```
   "status"  : 0
   ```
   or
   ```
   "status"  : 1
   "reason"  : <string>
   ```

2) Removal of an existing __object__

   _Client request_
   ```
   "request" : remove/session-id/object-id
   ```

   _Server response_
   ```
   "status"  : 0
   ```
   or
   ```
   "status"  : 1
   "reason"  : <string>
   ```

## Information retrieval commands - `get/`

1) Get all __session-ids__

   _Client request_
   ```
   "request" : get
   ```

   _Server response_
   ```
   "status"  : 0
   "data"    : { "ids" : [<comma-separated list of session ids>] }
   ```
   or
   ```
   "status"  : 1
   "reason"  : <string>
   ```

2) Get all __object-ids__ of a specific session

   _Client request_
   ```
   "request" : get/session-id
   ```

   _Server response_
   ```
   "status"  : 0
   "data"    : { "ids" : [<comma-separated list of object ids>] }
   ```
   or
   ```
   "status"  : 1
   "reason"  : <string>
   ```

3) Get _all_ __attributes__ of a specific object

   _Client request_
   ```
   "request" : get/session-id/object-id
   ```

   _Server response_
   ```
   "status"  : 0
   "data"    : {...}
   ```
   or
   ```
   "status"  : 1
   "reason"  : <string>
   ```

   Supported __object-types__:

   -   `uniformBSpline` and `nonuniformBSpline`
       ```
       "data" : { 
                  "geoDim"  : <integer> valid values are 1,2,3,4
                  "parDim"  : <integer> valid values are 1,2,3,4
                  "degrees" : [<comma-separated list of integers>]
                  "ncoeffs" : [<comma-separated list of integers>]
                  "nknots"  : [<comma-separated list of integers>]
                  "coeffs"  : [[comma-separated list of lists of floats]
                  "knots"   : [[comma-separated list of lists of floats]
                }
       ```

   -   More object types will be added

4) Get a _specific_ __attribute__ of a specific object

   _Client request_
   ```
   "request" : get/session-id/object-id/attribute
   ```
   The __attribute__ must be one of the supported attributes of the object type:

   -   `uniformBSpline` and `nonuniformBSpline`
       ```
       geoDim, parDim, degrees, ncoeffs, nknots, coeffs, knots
       ```

   _Server response_
   ```
   "status"  : 0
   "data"    : {...}
   ```
   or
   ```
   "status"  : 1
   "reason"  : <string>
   ```
   The __data__ will be formatted in the same format as in the _all_ __attribute__ case.

## Information update commands - `put/`

1) Update a _specific_ __attribute__ of a specific object

   _Client request_
   ```
   "request" : put/session-id/object-id/attribute
   ```
   Supported __attributes__:

   -   `uniformBSpline` and `nonuniformBSpline`
       ```
       "coeffs" : [[<comma-separated list of lists of floats>]]
       ```

   _Server response_
   ```
   "status"  : 0
   "data"    : {...}
   ```
   or
   ```
   "status"  : 1
   "reason"  : <string>
   ```

## Specialized command - `eval/`

   The __object-types__ `uniformBSpline` and `nonuniformBSpline` support the evaluation at discrete points in the parameter space

   _Client request_
   ```
   "request" : eval/session-id/object-id
   ```

   _Server response_
   ```
   "status"  : 0
   "data" : { 
              "coords" : [[<ist of lists of floats in lexicographical order>]]
              "values" : [<list of floats in lexicographical order>]
            }
   ```
   or
   ```
   "status"  : 1
   "reason"  : <string>
   ```

   ## Specialized command - `refine/`

   The __object-types__ `uniformBSpline` and `nonuniformBSpline` support regular refinement

   _Client request_
   ```
   "request" : refine/session-id/object-id
   "data"    : {
                 [ "numRefine" : <integer> (default value is 1) ]
                 [ "dim"       : <integer> (default value is -1) ]
               }
   ```
   If no __data__ field is provided the default values are adopted.

   _Server response_
   ```
   "status"  : 0
   ```
   or
   ```
   "status"  : 1
   "reason"  : <string>
   ```