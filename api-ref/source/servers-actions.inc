.. -*- rst -*-
.. needs:method_verification
.. needs:parameter_verification
.. needs:example_verification
.. needs:body_verification

===========================================
 Servers - run an action (servers, action)
===========================================

Enables all users to performs an action on a server. Specify the action
in the request body.

You can associate a fixed or floating IP address with a server instance,
or disassociate a fixed or floating IP address from a server instance.
You can attach a volume to a server.

You can create an image from a server, evacuate a server from a failed
host to a new host, and force-delete a server before deferred cleanup.
You can lock, pause, reboot, rebuild, rescue, resize, resume, confirm
the resize of, revert a pending resize for, shelve, shelf-offload,
unshelve, start, stop, unlock, unpause, and unrescue a server. You can
also change the password of the server and add a security group to or
remove a security group from a server. You can also trigger a crash dump
into a server since Mitaka release.

You can get an RDP, serial, SPICE, or VNC console for a server.

Add (Associate) Floating Ip (Addfloatingip Action)
==================================================

.. rest_method:: POST /v2.1/{tenant_id}/servers/{server_id}/action

Adds a floating IP address to a server, which associates that address with the server.

A pool of floating IP addresses, configured by the cloud administrator, is available in OpenStack Compute. The project quota defines the maximum number of floating IP addresses that you can allocate to the project. After you `create (allocate) a floating IPaddress <http://developer.openstack.org/api-ref-compute-v2.1.html#createFloatingIP>`__ for a project, you can associate that address with the server. Specify the ``addFloatingIp`` action in the request body.

If an instance is connected to multiple networks, you can associate a floating IP address with a specific fixed IP address by using the optional ``fixed_address`` parameter.

Preconditions

The server must exist.

You can only add a floating IP address to the server when its status is ``available``.

Normal response codes: 202

Error response codes: computeFault(400, 500), serviceUnavailable(503), badRequest(400),
unauthorized(401), forbidden(403), badMethod(405), itemNotFound(404)

Request
-------

.. rest_parameters:: parameters.yaml

  - tenant_id: tenant_id
  - server_id: server_id
  - addFloatingIp: addFloatingIp
  - address: address
  - fixed_address: fixed_address

.. TODO(gmann) Need to create the sample file for this action.
   **Example Add (Associate) Floating Ip (Addfloatingip Action): JSON request**

   .. literalinclude:: ../../doc/api_samples/os-floating-ips/floating-ips-create-resp.json
      :language: javascript

Response
--------

Add Security Group To A Server (Addsecuritygroup Action)
========================================================

.. rest_method:: POST /v2.1/{tenant_id}/servers/{server_id}/action

Adds a security group to a server.

Specify the ``addSecurityGroup`` action in the request body.

Normal response codes: 202

Error response codes: computeFault(400, 500), serviceUnavailable(503), badRequest(400),
unauthorized(401), forbidden(403), badMethod(405), itemNotFound(404)

Request
-------

.. rest_parameters:: parameters.yaml

  - tenant_id: tenant_id
  - server_id: server_id
  - addSecurityGroup: addSecurityGroup
  - name: name

**Example Add Security Group To A Server (Addsecuritygroup Action): JSON request**

.. literalinclude:: ../../doc/api_samples/os-security-groups/security-group-add-post-req.json
   :language: javascript

Response
--------

Change Administrative Password (Changepassword Action)
======================================================

.. rest_method:: POST /v2.1/{tenant_id}/servers/{server_id}/action

Changes the administrative password for a server.

Specify the ``changePassword`` action in the request body.

Policy defaults enable only users with the administrative role or the owner of the server to perform this operation. Cloud providers can change these permissions through the ``policy.json`` file.

Normal response codes: 202

Error response codes: computeFault(400, 500), serviceUnavailable(503), badRequest(400),
unauthorized(401), forbidden(403), badMethod(405), itemNotFound(404)

Request
-------

.. rest_parameters:: parameters.yaml

  - tenant_id: tenant_id
  - server_id: server_id
  - changePassword: changePassword

**Example Change Administrative Password (Changepassword Action): JSON request**

.. literalinclude:: ../../doc/api_samples/os-admin-password/admin-password-change-password.json
   :language: javascript

Response
--------

Confirm Resized Server (Confirmresize Action)
=============================================

.. rest_method:: POST /v2.1/{tenant_id}/servers/{server_id}/action

Confirms a pending resize action for a server.

Specify the ``confirmResize`` action in the request body.

After you make this request, you typically must keep polling the server status to determine whether the request succeeded. A successfully confirming resize operation shows a status of ``ACTIVE`` or ``SHUTOFF`` and a migration_status of ``confirmed``. You can also see the resized server in the compute node that OpenStack Compute manages.

Preconditions

You can only confirm the resized server where the status is ``VERIFY_RESIZED``, the vm_status is ``RESIZED``, and the migration_status is ``finished`` or ``confirming``.

If the server is locked, you must have administrator privileges to confirm the server.

Troubleshooting

If the server status remains ``RESIZED``, the request failed. Ensure you meet the preconditions and run the request again. If the request fails again, investigate the compute back end or ask your cloud provider.

Normal response codes: 204

Error response codes: computeFault(400, 500), serviceUnavailable(503), badRequest(400),
unauthorized(401), forbidden(403), badMethod(405), itemNotFound(404)

Request
-------

.. rest_parameters:: parameters.yaml

  - tenant_id: tenant_id
  - server_id: server_id
  - confirmResize: confirmResize

**Example Confirm Resized Server (Confirmresize Action): JSON request**

.. literalinclude:: ../../doc/api_samples/servers/server-action-confirm-resize.json
   :language: javascript

Response
--------

Create Image (Createimage Action)
=================================

.. rest_method:: POST /v2.1/{tenant_id}/servers/{server_id}/action

Creates an image from a server.

Specify the ``createImage`` action in the request body.

After you make this request, you typically must keep polling the status of the created image to determine whether the request succeeded.

If the operation succeeds, the created image has a status of ``active`` and the server status returns to the original status. You can also see the new image in the image back end that OpenStack Image service manages.

Preconditions

The server must exist.

You can only create a new image from the server when its status is ``ACTIVE``, ``SHUTOFF``, ``PAUSED``, or ``SUSPENDED``.

The connection to the Image service is valid.

Troubleshooting

If the image status remains uploading or shows another error status, the request failed. Ensure you meet the preconditions and run the request again. If the request fails again, investigate the image back end.

If the server status does not go back to an original server's status, the request failed. Ensure you meet the preconditions, or check if there is another operation that causes race conditions for the server, then run the request again. If the request fails again, investigate the compute back end or ask your cloud provider.

If the request fails due to an error on OpenStack Compute service, the image is purged from the image store that OpenStack Image service manages. Ensure you meet the preconditions and run the request again. If the request fails again, investigate OpenStack Compute service or ask your cloud provider.

Normal response codes: 202

Error response codes: computeFault(400, 500), serviceUnavailable(503), badRequest(400),
unauthorized(401), forbidden(403), badMethod(405), itemNotFound(404)

Request
-------

.. rest_parameters:: parameters.yaml

  - tenant_id: tenant_id
  - server_id: server_id
  - createImage: createImage

**Example Create Image (Createimage Action): JSON request**

.. literalinclude:: ../../doc/api_samples/servers/server-action-create-image.json
   :language: javascript

Response
--------

Lock Server (Lock Action)
=========================

.. rest_method:: POST /v2.1/{tenant_id}/servers/{server_id}/action

Locks a server.

Specify the ``lock`` action in the request body.

Policy defaults enable only users with the administrative role or the owner of the server to perform this operation. Cloud providers can change these permissions through the ``policy.json`` file.

Normal response codes: 202

Error response codes: computeFault(400, 500), serviceUnavailable(503), badRequest(400),
unauthorized(401), forbidden(403), badMethod(405), itemNotFound(404)

Request
-------

.. rest_parameters:: parameters.yaml

  - tenant_id: tenant_id
  - server_id: server_id
  - lock: lock

**Example Lock Server (Lock Action): JSON request**

.. literalinclude:: ../../doc/api_samples/os-lock-server/lock-server.json
   :language: javascript

Response
--------

Pause Server (Pause Action)
===========================

.. rest_method:: POST /v2.1/{tenant_id}/servers/{server_id}/action

Pauses a server. Changes its status to ``PAUSED``.

Specify the ``pause`` action in the request body.

Policy defaults enable only users with the administrative role or the owner of the server to perform this operation. Cloud providers can change these permissions through the ``policy.json`` file.

Normal response codes: 202

Error response codes: computeFault(400, 500), serviceUnavailable(503), badRequest(400),
unauthorized(401), forbidden(403), badMethod(405), itemNotFound(404)

Request
-------

.. rest_parameters:: parameters.yaml

  - tenant_id: tenant_id
  - server_id: server_id
  - pause: pause

**Example Pause Server (Pause Action): JSON request**

.. literalinclude:: ../../doc/api_samples/os-pause-server/pause-server.json
   :language: javascript

Response
--------

Reboot Server (Reboot Action)
=============================

.. rest_method:: POST /v2.1/{tenant_id}/servers/{server_id}/action

Reboots a server.

Specify the ``reboot`` action in the request body.

Normal response codes: 202

Error response codes: computeFault(400, 500), serviceUnavailable(503), badRequest(400),
unauthorized(401), forbidden(403), badMethod(405), itemNotFound(404)

Request
-------

.. rest_parameters:: parameters.yaml

  - tenant_id: tenant_id
  - server_id: server_id
  - reboot: reboot

**Example Reboot Server (Reboot Action): JSON request**

.. literalinclude:: ../../doc/api_samples/servers/server-action-reboot.json
   :language: javascript

Response
--------

Rebuild Server (Rebuild Action)
===============================

.. rest_method:: POST /v2.1/{tenant_id}/servers/{server_id}/action

Rebuilds a server.

Specify the ``rebuild`` action in the request body.

To rebuild the server with preservation of the ephemeral partition, set the ``preserve_ephemeral`` parameter to ``true``.

Normal response codes: 202

Error response codes: computeFault(400, 500), serviceUnavailable(503), badRequest(400),
unauthorized(401), forbidden(403), badMethod(405), itemNotFound(404)

Request
-------

.. rest_parameters:: parameters.yaml

  - tenant_id: tenant_id
  - server_id: server_id
  - rebuild: rebuild
  - imageRef: imageRef
  - name: name
  - adminPass: adminPass_evacuate_rebuild
  - metadata: metadata
  - personality: personality
  - preserve_ephemeral: preserve_ephemeral

**Example Rebuild Server (Rebuild Action): JSON request**

Preserving the ephemeral disk

.. literalinclude:: ../../doc/api_samples/servers/server-action-rebuild-preserve-ephemeral.json
   :language: javascript

Response
--------

**Example Rebuild Server (Rebuild Action): JSON response**

.. literalinclude:: ../../doc/api_samples/servers/server-action-rebuild-resp.json
   :language: javascript

Remove (Disassociate) Floating Ip (Removefloatingip Action)
===========================================================

.. rest_method:: POST /v2.1/{tenant_id}/servers/{server_id}/action

Removes, or disassociates, a floating IP address from a server.

The IP address is returned to the pool of IP addresses that is available for all projects. When you remove a floating IP address and that IP address is still associated with a running instance, it is automatically disassociated from that instance.

Specify the ``removeFloatingIp`` action in the request body.

Normal response codes: 202

Error response codes: computeFault(400, 500), serviceUnavailable(503), badRequest(400),
unauthorized(401), forbidden(403), badMethod(405), itemNotFound(404)

Request
-------

.. rest_parameters:: parameters.yaml

  - tenant_id: tenant_id
  - server_id: server_id
  - removeFloatingIp: removeFloatingIp
  - address: address

.. TODO(sdague) build example for this
   **Example Remove (Disassociate) Floating Ip (Removefloatingip Action): JSON request**

   .. literalinclude:: ../../doc/api_samples/servers-action/removeFloatingIp-req.json
                       :language: javascript

Response
--------

Remove Security Group From A Server (Removesecuritygroup Action)
================================================================

.. rest_method:: POST /v2.1/{tenant_id}/servers/{server_id}/action

Removes a security group from a server.

Specify the ``removeSecurityGroup`` action in the request body.

Normal response codes: 202

Error response codes: computeFault(400, 500), serviceUnavailable(503), badRequest(400),
unauthorized(401), forbidden(403), badMethod(405), itemNotFound(404)

Request
-------

.. rest_parameters:: parameters.yaml

  - tenant_id: tenant_id
  - server_id: server_id
  - removeSecurityGroup: removeSecurityGroup
  - name: name

**Example Remove Security Group From A Server (Removesecuritygroup Action): JSON request**

.. literalinclude:: ../../doc/api_samples/os-security-groups/security-group-remove-post-req.json
   :language: javascript

Response
--------

Rescue Server (Rescue Action)
=============================

.. rest_method:: POST /v2.1/{tenant_id}/servers/{server_id}/action

Puts a server in rescue mode and changes its status to ``RESCUE``.

Specify the ``rescue`` action in the request body.

If you specify the ``rescue_image_ref`` extended attribute, the image is used to rescue the instance. If you omit an image reference, the base image reference is used by default.

Asynchronous Postconditions

After you successfully rescue a server and make a ``GET /v2.1/​{tenant_id}​/servers/​{server_id}​`` request, its status changes to ``RESCUE``.

Normal response codes: 202

Error response codes: computeFault(400, 500), serviceUnavailable(503), badRequest(400),
unauthorized(401), forbidden(403), badMethod(405), itemNotFound(404)

Request
-------

.. rest_parameters:: parameters.yaml

  - tenant_id: tenant_id
  - server_id: server_id
  - rescue: rescue
  - adminPass: adminPass_rescue
  - rescue_image_ref: rescue_image_ref

**Example Rescue server: JSON request**

.. literalinclude:: ../../doc/api_samples/os-rescue/server-rescue-req.json
   :language: javascript

Response
--------

.. rest_parameters:: parameters.yaml

  - adminPass: adminPass

**Example Extended rescue server: JSON response**

.. literalinclude:: ../../doc/api_samples/os-rescue/server-rescue.json
   :language: javascript

Resize Server (Resize Action)
=============================

.. rest_method:: POST /v2.1/{tenant_id}/servers/{server_id}/action

Resizes a server.

Specify the ``resize`` action in the request body.

A successfully resized server shows a ``VERIFY_RESIZE`` status, ``RESIZED`` VM status, and ``finished`` migration status. If you set the ``resize_confirm_window`` option of the Compute service to an integer value, the Compute service automatically confirms the resize operation after the set interval in seconds.

Preconditions

You can only resize a server when its status is ``ACTIVE`` or ``SHUTOFF``.

If the server is locked, you must have administrator privileges to resize the server.

Normal response codes: 202

Error response codes: computeFault(400, 500), serviceUnavailable(503), badRequest(400),
unauthorized(401), forbidden(403), badMethod(405), itemNotFound(404)

Request
-------

.. rest_parameters:: parameters.yaml

  - tenant_id: tenant_id
  - server_id: server_id
  - resize: resize

**Example Resize Server (Resize Action): JSON request**

.. literalinclude:: ../../doc/api_samples/servers/server-action-resize.json
   :language: javascript

Response
--------

Resume Suspended Server (Resume Action)
=======================================

.. rest_method:: POST /v2.1/{tenant_id}/servers/{server_id}/action

Resumes a suspended server and changes its status to ``ACTIVE``.

Specify the ``resume`` action in the request body.

Policy defaults enable only users with the administrative role or the owner of the server to perform this operation. Cloud providers can change these permissions through the ``policy.json`` file.

Normal response codes: 202

Error response codes: computeFault(400, 500), serviceUnavailable(503), badRequest(400),
unauthorized(401), forbidden(403), badMethod(405), itemNotFound(404)

Request
-------

.. rest_parameters:: parameters.yaml

  - tenant_id: tenant_id
  - server_id: server_id
  - resume: resume

**Example Resume Suspended Server (Resume Action): JSON request**

.. literalinclude:: ../../doc/api_samples/os-suspend-server/server-resume.json
   :language: javascript

Response
--------

Revert Resized Server (Revertresize Action)
===========================================

.. rest_method:: POST /v2.1/{tenant_id}/servers/{server_id}/action

Cancels and reverts a pending resize action for a server.

Specify the ``revertResize`` action in the request body.

After you make this request, you typically must keep polling the server status to determine whether the request succeeded. A successfully reverting resize operation shows a status of ``ACTIVE`` or ``SHUTOFF`` and a migration_status of ``reverted``. You can also see the reverted server in the compute node that OpenStack Compute manages.

Preconditions

You can only confirm the resized server where the status is ``VERIFY_RESIZE`` and the vm_status is ``RESIZED``.

If the server is locked, you must have administrator privileges to revert the resizing.

Troubleshooting

If the server status remains ``RESIZED``, the request failed. Ensure you meet the preconditions and run the request again. If the request fails again, investigate the compute back end.

The server is not reverted in the compute node that OpenStack Compute manages.

Normal response codes: 202

Error response codes: computeFault(400, 500), serviceUnavailable(503), badRequest(400),
unauthorized(401), forbidden(403), badMethod(405), itemNotFound(404)

Request
-------

.. rest_parameters:: parameters.yaml

  - tenant_id: tenant_id
  - server_id: server_id
  - revertResize: revertResize

**Example Revert Resized Server (Revertresize Action): JSON request**

.. literalinclude:: ../../doc/api_samples/servers/server-action-revert-resize.json
   :language: javascript

Response
--------

Start Server (Os-Start Action)
==============================

.. rest_method:: POST /v2.1/{tenant_id}/servers/{server_id}/action

Starts a stopped server and changes its status to ``ACTIVE``.

Specify the ``os-start`` action in the request body.

Preconditions

The server status must be ``SHUTOFF``.

If the server is locked, you must have administrator privileges to start the server.

Asynchronous Postconditions

After you successfully start a server, its status changes to ``ACTIVE``. The server appears on the compute node that the Compute service manages.

Troubleshooting

If the server status does not change to ``ACTIVE``, the start operation failed. Ensure that you meet the preconditions and run the request again. If the request fails again, investigate whether another operation is running that causes a race condition.

Normal response codes: 202

Error response codes: computeFault(400, 500), serviceUnavailable(503), badRequest(400),
unauthorized(401), forbidden(403), badMethod(405), itemNotFound(404)

Request
-------

.. rest_parameters:: parameters.yaml

  - tenant_id: tenant_id
  - server_id: server_id
  - os-start: os-start

**Example Start server: JSON request**

.. literalinclude:: ../../doc/api_samples/servers/server-action-start.json
   :language: javascript

Response
--------

Stop Server (Os-Stop Action)
============================

.. rest_method:: POST /v2.1/{tenant_id}/servers/{server_id}/action

Stops a running server and changes its status to ``SHUTOFF``.

Specify the ``os-stop`` action in the request body.

Preconditions

The server status must be ``ACTIVE`` or ``ERROR``.

If the server is locked, you must have administrator privileges to stop the server.

Asynchronous Postconditions

After you successfully stop a server, its status changes to ``SHUTOFF``. The server instance data appears only on the compute node that Compute service manages.

Normal response codes: 202

Error response codes: computeFault(400, 500), serviceUnavailable(503), badRequest(400),
unauthorized(401), forbidden(403), badMethod(405), itemNotFound(404)

Request
-------

.. rest_parameters:: parameters.yaml

  - tenant_id: tenant_id
  - server_id: server_id
  - os-stop: os-stop

**Example Stop server: JSON request**

.. literalinclude:: ../../doc/api_samples/servers/server-action-stop.json
   :language: javascript

Response
--------

Suspend Server (Suspend Action)
===============================

.. rest_method:: POST /v2.1/{tenant_id}/servers/{server_id}/action

Suspends a server and changes its status to ``SUSPENDED``.

Specify the ``suspend`` action in the request body.

Policy defaults enable only users with the administrative role or the owner of the server to perform this operation. Cloud providers can change these permissions through the ``policy.json`` file.

Normal response codes: 202

Error response codes: computeFault(400, 500), serviceUnavailable(503), badRequest(400),
unauthorized(401), forbidden(403), badMethod(405), itemNotFound(404)

Request
-------

.. rest_parameters:: parameters.yaml

  - tenant_id: tenant_id
  - server_id: server_id
  - suspend: suspend

**Example Suspend Server (Suspend Action): JSON request**

.. literalinclude:: ../../doc/api_samples/os-suspend-server/server-suspend.json
   :language: javascript

Response
--------

Unlock Server (Unlock Action)
=============================

.. rest_method:: POST /v2.1/{tenant_id}/servers/{server_id}/action

Unlocks a locked server.

Specify the ``unlock`` action in the request body.

Policy defaults enable only users with the administrative role or the owner of the server to perform this operation. Cloud providers can change these permissions through the ``policy.json`` file.

Normal response codes: 202

Error response codes: computeFault(400, 500), serviceUnavailable(503), badRequest(400),
unauthorized(401), forbidden(403), badMethod(405), itemNotFound(404)

Request
-------

.. rest_parameters:: parameters.yaml

  - tenant_id: tenant_id
  - server_id: server_id
  - unlock: unlock

**Example Unlock Server (Unlock Action): JSON request**

.. literalinclude:: ../../doc/api_samples/os-lock-server/unlock-server.json
   :language: javascript

Response
--------

Unpause Server (Unpause Action)
===============================

.. rest_method:: POST /v2.1/{tenant_id}/servers/{server_id}/action

Unpauses a paused server and changes its status to ``ACTIVE``.

Specify the ``unpause`` action in the request body.

Policy defaults enable only users with the administrative role or the owner of the server to perform this operation. Cloud providers can change these permissions through the ``policy.json`` file.

Normal response codes: 202

Error response codes: computeFault(400, 500), serviceUnavailable(503), badRequest(400),
unauthorized(401), forbidden(403), badMethod(405), itemNotFound(404)

Request
-------

.. rest_parameters:: parameters.yaml

  - tenant_id: tenant_id
  - server_id: server_id
  - unpause: unpause

**Example Unpause Server (Unpause Action): JSON request**

.. literalinclude:: ../../doc/api_samples/os-pause-server/unpause-server.json
   :language: javascript

Response
--------

Unrescue Server (Unrescue Action)
=================================

.. rest_method:: POST /v2.1/{tenant_id}/servers/{server_id}/action

Unrescues a server. Changes status to ``ACTIVE``.

Specify the ``unrescue`` action in the request body.

Preconditions

The server must exist.

You can only unrescue a server when its status is ``RESCUE``.

Asynchronous Postconditions

After you successfully unrescue a server and make a ``GET /v2.1/​{tenant_id}​/servers/​{server_id}​`` request, its status changes to ``ACTIVE``.

Normal response codes: 202

Error response codes: computeFault(400, 500), serviceUnavailable(503), badRequest(400),
unauthorized(401), forbidden(403), badMethod(405), itemNotFound(404)

Request
-------

.. rest_parameters:: parameters.yaml

  - tenant_id: tenant_id
  - server_id: server_id
  - unrescue: unrescue

**Example Unrescue server: JSON request**

.. literalinclude:: ../../doc/api_samples/os-rescue/server-unrescue-req.json
   :language: javascript

Response
--------
