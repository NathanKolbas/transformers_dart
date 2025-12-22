//
//  Generated file. Do not edit.
//

// clang-format off

#include "generated_plugin_registrant.h"

#include <transformers/transformers_plugin.h>

void fl_register_plugins(FlPluginRegistry* registry) {
  g_autoptr(FlPluginRegistrar) transformers_registrar =
      fl_plugin_registry_get_registrar_for_plugin(registry, "TransformersPlugin");
  transformers_plugin_register_with_registrar(transformers_registrar);
}
