# Copyright 2012 Michael Still and Canonical Inc
# All Rights Reserved.
#
#    Licensed under the Apache License, Version 2.0 (the "License"); you may
#    not use this file except in compliance with the License. You may obtain
#    a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#    WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#    License for the specific language governing permissions and limitations
#    under the License.

"""Image cache manager.

The cache manager implements the specification at
http://wiki.openstack.org/nova-image-cache-management.

"""
import collections
import hashlib
import os
import re
import time

from oslo_concurrency import lockutils
from oslo_concurrency import processutils
from oslo_log import log as logging
from oslo_utils import fileutils

import nova.conf
from nova import exception
from nova.i18n import _LE
from nova.i18n import _LI
from nova.i18n import _LW
from nova import utils
from nova.virt import imagecache
from nova.virt import images
from nova.virt.libvirt import utils as libvirt_utils

LOG = logging.getLogger(__name__)

CONF = nova.conf.CONF


def get_cache_fname(image_id):
    """Return a filename based on the SHA1 hash of a given image ID.

    Image files stored in the _base directory that match this pattern
    are considered for cleanup by the image cache manager. The cache
    manager considers the file to be in use if it matches an instance's
    image_ref, kernel_id or ramdisk_id property.
    """
    return hashlib.sha1(image_id.encode('utf-8')).hexdigest()


def get_info_filename(base_path):
    """Construct a filename for storing additional information about a base
    image.

    Returns a filename.
    """

    base_file = os.path.basename(base_path)
    return (CONF.libvirt.image_info_filename_pattern
            % {'image': base_file})


def is_valid_info_file(path):
    """Test if a given path matches the pattern for info files."""

    digest_size = hashlib.sha1().digestsize * 2
    regexp = (CONF.libvirt.image_info_filename_pattern
              % {'image': ('([0-9a-f]{%(digest_size)d}|'
                           '[0-9a-f]{%(digest_size)d}_sm|'
                           '[0-9a-f]{%(digest_size)d}_[0-9]+)'
                           % {'digest_size': digest_size})})
    m = re.match(regexp, path)
    if m:
        return True
    return False


def lock_path():
    return os.path.join(CONF.instances_path, 'locks')


def locked_cache_entry(name):
    return utils.synchronized(name, external=True, lock_path=lock_path())


def delete_lock_file(name):
    lockutils.remove_external_lock_file(name,
        lock_file_prefix='nova-', lock_path=lock_path())


class ImageCacheManager(imagecache.ImageCacheManager):
    def __init__(self):
        super(ImageCacheManager, self).__init__()
        self._reset_state()

    def _reset_state(self):
        """Reset state variables used for each pass."""

        self.used_images = {}
        self.image_popularity = {}
        self.instance_names = set()

        self.back_swap_images = set()
        self.used_swap_images = set()

        self.active_base_files = []
        self.originals = []
        self.removable_base_files = []
        self.unexplained_images = []

    def _store_image(self, base_dir, ent, original=False):
        """Store a base image for later examination."""
        entpath = os.path.join(base_dir, ent)
        if os.path.isfile(entpath):
            self.unexplained_images.append(entpath)
            if original:
                self.originals.append(entpath)

    def _store_swap_image(self, ent):
        """Store base swap images for later examination."""
        names = ent.split('_')
        if len(names) == 2 and names[0] == 'swap':
            if len(names[1]) > 0 and names[1].isdigit():
                LOG.debug('Adding %s into backend swap images', ent)
                self.back_swap_images.add(ent)

    def _list_base_images(self, base_dir):
        """Return a list of the images present in _base.

        Determine what images we have on disk. There will be other files in
        this directory so we only grab the ones which are the right length
        to be disk images.
        """

        digest_size = hashlib.sha1().digestsize * 2
        for ent in os.listdir(base_dir):
            path = os.path.join(base_dir, ent)
            if is_valid_info_file(path):
                # TODO(mdbooth): In Newton we ignore these files, because if
                # we're on shared storage they may be in use by a pre-Newton
                # compute host. However, we have already removed all uses of
                # these files in Newton, so once we can be sure that all
                # compute hosts are running at least Newton (i.e. in  Ocata),
                # we can be sure that nothing is using info files any more.
                # Therefore in Ocata, we should update this to simply delete
                # these files here, i.e.:
                #   os.unlink(path)
                #
                # This will obsolete the code to cleanup these files in
                # _remove_old_enough_file, so when updating this code to
                # delete immediately, the cleanup code in
                # _remove_old_enough_file can be removed.
                #
                # This cleanup code will delete all info files the first
                # time it runs in Ocata, which means we can delete this
                # block entirely in P.
                pass

            elif len(ent) == digest_size:
                self._store_image(base_dir, ent, original=True)

            elif len(ent) > digest_size + 2 and ent[digest_size] == '_':
                self._store_image(base_dir, ent, original=False)

            else:
                self._store_swap_image(ent)

        return {'unexplained_images': self.unexplained_images,
                'originals': self.originals}

    def _list_backing_images(self):
        """List the backing images currently in use."""
        inuse_images = []
        for ent in os.listdir(CONF.instances_path):
            if ent in self.instance_names:
                LOG.debug('%s is a valid instance name', ent)
                disk_path = os.path.join(CONF.instances_path, ent, 'disk')
                if os.path.exists(disk_path):
                    LOG.debug('%s has a disk file', ent)
                    try:
                        backing_file = libvirt_utils.get_disk_backing_file(
                            disk_path)
                    except processutils.ProcessExecutionError:
                        # (for bug 1261442)
                        if not os.path.exists(disk_path):
                            LOG.debug('Failed to get disk backing file: %s',
                                      disk_path)
                            continue
                        else:
                            raise
                    LOG.debug('Instance %(instance)s is backed by '
                              '%(backing)s',
                              {'instance': ent,
                               'backing': backing_file})

                    if backing_file:
                        backing_path = os.path.join(
                            CONF.instances_path,
                            CONF.image_cache_subdirectory_name,
                            backing_file)
                        if backing_path not in inuse_images:
                            inuse_images.append(backing_path)

                        if backing_path in self.unexplained_images:
                            LOG.warning(_LW('Instance %(instance)s is using a '
                                         'backing file %(backing)s which '
                                         'does not appear in the image '
                                         'service'),
                                        {'instance': ent,
                                         'backing': backing_file})
                            self.unexplained_images.remove(backing_path)
        return inuse_images

    def _find_base_file(self, base_dir, fingerprint):
        """Find the base file matching this fingerprint.

        Yields the name of the base file, a boolean which is True if the image
        is "small", and a boolean which indicates if this is a resized image.
        Note that it is possible for more than one yield to result from this
        check.

        If no base file is found, then nothing is yielded.
        """
        # The original file from glance
        base_file = os.path.join(base_dir, fingerprint)
        if os.path.exists(base_file):
            yield base_file, False, False

        # An older naming style which can be removed sometime after Folsom
        base_file = os.path.join(base_dir, fingerprint + '_sm')
        if os.path.exists(base_file):
            yield base_file, True, False

        # Resized images
        resize_re = re.compile('.*/%s_[0-9]+$' % fingerprint)
        for img in self.unexplained_images:
            m = resize_re.match(img)
            if m:
                yield img, False, True

    @staticmethod
    def _get_age_of_file(base_file):
        if not os.path.exists(base_file):
            LOG.debug('Cannot remove %s, it does not exist', base_file)
            return (False, 0)

        mtime = os.path.getmtime(base_file)
        age = time.time() - mtime

        return (True, age)

    def _remove_old_enough_file(self, base_file, maxage, remove_lock=True):
        """Remove a single swap or base file if it is old enough."""
        exists, age = self._get_age_of_file(base_file)
        if not exists:
            return

        lock_file = os.path.split(base_file)[-1]

        @locked_cache_entry(lock_file)
        def _inner_remove_old_enough_file():
            # NOTE(mikal): recheck that the file is old enough, as a new
            # user of the file might have come along while we were waiting
            # for the lock
            exists, age = self._get_age_of_file(base_file)
            if not exists or age < maxage:
                return

            LOG.info(_LI('Removing base or swap file: %s'), base_file)
            try:
                os.remove(base_file)

                # TODO(mdbooth): We have removed all uses of info files in
                # Newton and we no longer create them, but they may still
                # exist from before we upgraded, and they may still be
                # created by older compute hosts if we're on shared storage.
                # While there may still be pre-Newton computes writing here,
                # the only safe place to delete info files is here,
                # when deleting the cache entry. Once we can be sure that
                # all computes are running at least Newton (i.e. in Ocata),
                # we can delete these files unconditionally during the
                # periodic task, which will make this code obsolete.
                signature = get_info_filename(base_file)
                if os.path.exists(signature):
                    os.remove(signature)
            except OSError as e:
                LOG.error(_LE('Failed to remove %(base_file)s, '
                              'error was %(error)s'),
                          {'base_file': base_file,
                           'error': e})

        if age < maxage:
            LOG.info(_LI('Base or swap file too young to remove: %s'),
                         base_file)
        else:
            _inner_remove_old_enough_file()
            if remove_lock:
                try:
                    # NOTE(mdbooth): This is a bug. Consider the following
                    # scenario:
                    #
                    # Thread A          Thread B        Thread C
                    # Get lock X        Get lock X
                    # Delete lock X
                    # Release lock X
                    #                   Obtain lock X   Get lock X'
                    #                   Download        Download
                    #
                    # Note that because we deleted the lock file thread B
                    # obtains a lock on the now-deleted file, and thread C
                    # obtains a different lock on a newly created lock file.
                    # Both have a lock simultaneously, and both download to
                    # the same destination simultaneously, creating a
                    # corrupt downloaded image.
                    #
                    # TODO(mdbooth): fix this bug
                    delete_lock_file(lock_file)
                except OSError as e:
                    LOG.debug('Failed to remove %(lock_file)s, '
                              'error was %(error)s',
                              {'lock_file': lock_file,
                               'error': e})

    def _remove_swap_file(self, base_file):
        """Remove a single swap base file if it is old enough."""
        maxage = CONF.remove_unused_original_minimum_age_seconds

        self._remove_old_enough_file(base_file, maxage, remove_lock=False)

    def _remove_base_file(self, base_file):
        """Remove a single base file if it is old enough."""
        maxage = CONF.libvirt.remove_unused_resized_minimum_age_seconds
        if base_file in self.originals:
            maxage = CONF.remove_unused_original_minimum_age_seconds

        self._remove_old_enough_file(base_file, maxage)

    def _handle_base_image(self, img_id, base_file):
        """Handle the checks for a single base image."""

        image_in_use = False

        LOG.info(_LI('image %(id)s at (%(base_file)s): checking'),
                 {'id': img_id,
                  'base_file': base_file})

        if base_file in self.unexplained_images:
            self.unexplained_images.remove(base_file)

        if img_id in self.used_images:
            local, remote, instances = self.used_images[img_id]

            if local > 0 or remote > 0:
                image_in_use = True
                LOG.info(_LI('image %(id)s at (%(base_file)s): '
                             'in use: on this node %(local)d local, '
                             '%(remote)d on other nodes sharing this instance '
                             'storage'),
                         {'id': img_id,
                          'base_file': base_file,
                          'local': local,
                          'remote': remote})

                self.active_base_files.append(base_file)

                if not base_file:
                    LOG.warning(_LW('image %(id)s at (%(base_file)s): warning '
                                 '-- an absent base file is in use! '
                                 'instances: %(instance_list)s'),
                                {'id': img_id,
                                 'base_file': base_file,
                                 'instance_list': ' '.join(instances)})

        if base_file:
            if not image_in_use:
                LOG.debug('image %(id)s at (%(base_file)s): image is not in '
                          'use',
                          {'id': img_id,
                           'base_file': base_file})
                self.removable_base_files.append(base_file)

            else:
                LOG.debug('image %(id)s at (%(base_file)s): image is in '
                          'use',
                          {'id': img_id,
                           'base_file': base_file})
                if os.path.exists(base_file):
                    libvirt_utils.update_mtime(base_file)

    def _age_and_verify_swap_images(self, context, base_dir):
        LOG.debug('Verify swap images')

        for ent in self.back_swap_images:
            base_file = os.path.join(base_dir, ent)
            if ent in self.used_swap_images and os.path.exists(base_file):
                libvirt_utils.update_mtime(base_file)
            elif self.remove_unused_base_images:
                self._remove_swap_file(base_file)

        error_images = self.used_swap_images - self.back_swap_images
        for error_image in error_images:
            LOG.warning(_LW('%s swap image was used by instance'
                         ' but no back files existing!'), error_image)

    def _age_and_verify_cached_images(self, context, all_instances, base_dir):
        LOG.debug('Verify base images')
        # Determine what images are on disk because they're in use
        for img in self.used_images:
            fingerprint = hashlib.sha1(img).hexdigest()
            LOG.debug('Image id %(id)s yields fingerprint %(fingerprint)s',
                      {'id': img,
                       'fingerprint': fingerprint})
            for result in self._find_base_file(base_dir, fingerprint):
                base_file, image_small, image_resized = result
                self._handle_base_image(img, base_file)

                if not image_small and not image_resized:
                    self.originals.append(base_file)

        # Elements remaining in unexplained_images might be in use
        inuse_backing_images = self._list_backing_images()
        for backing_path in inuse_backing_images:
            if backing_path not in self.active_base_files:
                self.active_base_files.append(backing_path)

        # Anything left is an unknown base image
        for img in self.unexplained_images:
            LOG.warning(_LW('Unknown base file: %s'), img)
            self.removable_base_files.append(img)

        # Dump these lists
        if self.active_base_files:
            LOG.info(_LI('Active base files: %s'),
                     ' '.join(self.active_base_files))

        if self.removable_base_files:
            LOG.info(_LI('Removable base files: %s'),
                     ' '.join(self.removable_base_files))

            if self.remove_unused_base_images:
                for base_file in self.removable_base_files:
                    self._remove_base_file(base_file)

        # That's it
        LOG.debug('Verification complete')

    def _get_base(self):

        # NOTE(mikal): The new scheme for base images is as follows -- an
        # image is streamed from the image service to _base (filename is the
        # sha1 hash of the image id). If CoW is enabled, that file is then
        # resized to be the correct size for the instance (filename is the
        # same as the original, but with an underscore and the resized size
        # in bytes). This second file is then CoW'd to the instance disk. If
        # CoW is disabled, the resize occurs as part of the copy from the
        # cache to the instance directory. Files ending in _sm are no longer
        # created, but may remain from previous versions.

        base_dir = os.path.join(CONF.instances_path,
                                CONF.image_cache_subdirectory_name)
        if not os.path.exists(base_dir):
            LOG.debug('Skipping verification, no base directory at %s',
                      base_dir)
            return
        return base_dir

    def update(self, context, all_instances):
        base_dir = self._get_base()
        if not base_dir:
            return
        # reset the local statistics
        self._reset_state()
        # read the cached images
        self._list_base_images(base_dir)
        # read running instances data
        running = self._list_running_instances(context, all_instances)
        self.used_images = running['used_images']
        self.image_popularity = running['image_popularity']
        self.instance_names = running['instance_names']
        self.used_swap_images = running['used_swap_images']
        # perform the aging and image verification
        self._age_and_verify_cached_images(context, all_instances, base_dir)
        self._age_and_verify_swap_images(context, base_dir)


CachedImageInfo = collections.namedtuple(
    'CachedImageInfo', ('path', 'file_format', 'virtual_size', 'disk_size'))


class ImageCacheLocalDir(object):
    """Cache glance images or template function output in a directory local
    to the compute host. This class provides an interface to write to and
    retrieve from the cache managed by ImageCacheManager.
    """

    # NOTE(mdbooth): Ideally the functionality of ImageCacheManager will
    # be folded into this class as it is cleaned up. For the moment, this is
    # the code which was distributed through other modules for interoperating
    # compatibly with ImageCacheManager.

    _singleton = None

    @classmethod
    def get(cls):
        """Get the ImageCacheLocalDir singleton.

        Returns:
            ImageCacheLocalDir: ImageCacheLocalDir singleton
        """
        if cls._singleton is not None:
            return cls._singleton

        with lockutils.lock('libvirt.imagebackend.imagecache_local_pool'):
            if cls._singleton is None:
                cls._singleton = ImageCacheLocalDir()
            return cls._singleton

    def __init__(self):
        self.lock_path = os.path.join(CONF.instances_path, 'locks')
        self.base_dir = os.path.join(CONF.instances_path,
                                     CONF.image_cache_subdirectory_name)
        if not os.path.exists(self.base_dir):
            fileutils.ensure_tree(self.base_dir)

    def is_path_in_image_cache(self, path):
        return os.path.dirname(path) == self.base_dir

    @staticmethod
    def _check_exists_and_mark_in_use(path):
        if os.path.exists(path):
            # NOTE(mikal): Update the mtime of the base file so the image
            # cache manager knows it is in use.
            libvirt_utils.update_mtime(path)
            return True

        return False

    def get_image_info(self, context, image_id, fallback_from_host=None):
        """Fetch an image from the cache, side-loading it from
        `fallback_from_host` if it isn't in glance.

        Args:
            context: The current RequestContext.
            image_id: The image_id of the image to be fetched.
            fallback_from_host: A compute host to side-load the image from
                    if it is no longer in glance.

        Returns:
            CachedImageInfo: Metadata describing the cached image.
        """
        name = get_cache_fname(image_id)
        path = os.path.join(self.base_dir, name)

        @locked_cache_entry(name)
        def _sync():
            if self._check_exists_and_mark_in_use(path):
                return

            try:
                libvirt_utils.fetch_image(context, path, image_id)
            except exception.ImageNotFound:
                if fallback_from_host is None:
                    raise

                LOG.debug("Image %(image_id)s no longer exists in the image "
                          "service. Attempting to copy it from %(host)s",
                          {'image_id': image_id, 'host': fallback_from_host})

                libvirt_utils.copy_image(src=path, dest=path,
                                         host=fallback_from_host,
                                         receive=True)

        _sync()

        # NOTE: We should persist this information somehow and return it
        # securely, because image inspection here isn't ideal. The image was
        # sanity checked in fetch_image when it was downloaded, so this
        # isn't the worst, but we could do better. Until then, this is
        # equivalent to what we have done before.
        info = images.qemu_img_info(path)
        return CachedImageInfo(path, info.file_format, info.virtual_size,
                               info.disk_size)

    def get_func_output_path(self, func, name, fallback_from_host=None):
        """Get the cached output of a function from the cache. If it
        don't exist, try to side-load it from another compute host.
        Failing that, generate it again locally.
        Args:
            func: A function, taking a path as an argument, which will write
                data to path
            name: The name of the function output, used to reference cached
                output.
            fallback_from_host: A compute host to side-load the output from.
        Returns:
            str: The path of the cached output.
        """
        path = os.path.join(self.base_dir, name)

        @locked_cache_entry(name)
        def _sync():
            if self._check_exists_and_mark_in_use(path):
                return

            created = False
            if fallback_from_host is not None:
                try:
                    # Ideally we'll get the original backing file from the
                    # source host. The reason we prefer this is that if we
                    # have to generate a new one, or if the output already
                    # existed in our own cache from a previous run,
                    # for that matter, the output is guaranteed to be
                    # different. This is a bug which results in (hopefully
                    # only) subtle data corruption.

                    LOG.debug("Disk template %(name)s doesn't exist in the "
                              "image cache, attempting to copy it from "
                              "%(host)s",
                              {'name': name, 'host': fallback_from_host})

                    libvirt_utils.copy_image(src=path, dest=path,
                                             host=fallback_from_host,
                                             receive=True)
                    created = True
                except processutils.ProcessExecutionError:
                    LOG.exception(_LW('Failed to side-load %(path)s from '
                                      '%(host)s'),
                                  {'path': path, 'host': fallback_from_host})

            # Generate a new one as a last resort
            if not created:
                func(path)

        _sync()

        return path
