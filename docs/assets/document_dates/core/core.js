const defaultConfig = {
    // configurable: light material, or custom theme in user.config.css
    theme: {
        light: 'light',
        dark: 'material'
    },
    tooltip: {
        placement: 'bottom',    // placement: top bottom left right auto
        offset: [0, 10],        // placement offset: [horizontal, vertical]
        interactive: true,      // content in Tooltip is interactive
        allowHTML: true,        // whether to allow HTML in the tooltip content
        
        animation: 'scale',     // animation type: scale shift-away
        inertia: true,          // animation inertia
        // arrow: false,           // whether to allow arrows

        // animateFill: true,      // determines if the background fill color should be animated

        // delay: [400, null],     // delay: [show, hide], show delay is 400ms, hide delay is the default
    }
};

let tippy_config = { ...defaultConfig };

// Configuration API
function setConfig(newConfig) {
    tippy_config = {
        ...defaultConfig,
        ...newConfig
    };
}



// Hook System
const hooks = {
    beforeInit: [],
    afterInit: []
};

// Hook registration API
function registerHook(hookName, callback) {
    if (hooks[hookName]) {
        hooks[hookName].push(callback);
    }
}

// Hook execution
async function executeHooks(hookName, context) {
    if (hooks[hookName]) {
        for (const hook of hooks[hookName]) {
            await hook(context);
        }
    }
}

// Theme management
function getCurrentTheme() {
    const scheme = (document.body && document.body.getAttribute('data-md-color-scheme')) || 'default';
    return scheme === 'slate' ? tippy_config.theme.dark : tippy_config.theme.light;
}

// Main initialization
async function init() {
    await executeHooks('beforeInit', { tippy_config });

    // Configure the properties of the Tooltip here, available documents: https://atomiks.github.io/tippyjs/
    const tippyInstances = tippy('[data-tippy-content]', {
        ...tippy_config.tooltip,
        theme: getCurrentTheme()    // Initialize Tooltip's theme based on Material's light/dark color scheme
    });

    // Automatic theme switching. Set Tooltip's theme to change automatically with the Material's light/dark color scheme
    const observer = new MutationObserver((mutations) => {
        mutations.forEach(async (mutation) => {
            if (mutation.attributeName === 'data-md-color-scheme') {
                const newTheme = getCurrentTheme();
                tippyInstances.forEach(instance => {
                    instance.setProps({ theme: newTheme });
                });
            }
        });
    });

    observer.observe(document.body, {
        attributes: true,
        attributeFilter: ['data-md-color-scheme']
    });

    await executeHooks('afterInit', { tippyInstances });
}

// Singleton Initialization
const initManager = (() => {
    let initialized = false;
    return {
        initialize() {
            if (initialized) return;
            init();
            initialized = true;
        }
    };
})();

// Entrance
document.addEventListener('DOMContentLoaded', initManager.initialize);



// Export API
window.DocumentDates = {
    registerHook,
    setConfig
};
