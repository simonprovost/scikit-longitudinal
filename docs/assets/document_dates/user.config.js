/*
    Configuration Overrides for Tooltip
    see: https://atomiks.github.io/tippyjs/
*/

/* Configure one by one: */
// tippy_config.theme.light = 'tomato';
// tippy_config.tooltip.placement = 'top';
// tippy_config.tooltip.arrow = false;

/* Or, override all configuration items: */
/*
DocumentDates.setConfig({
    theme: {
        // configurable: light material, or custom theme in user.config.css, for example: sorrel sublime tomato
        light: 'light',
        dark: 'material'
    },
    tooltip: {
        placement: 'bottom',    // placement: top bottom left right auto
        offset: [0, 6],         // placement offset: [horizontal, vertical]
        interactive: true,      // content in Tooltip is interactive
        allowHTML: true,        // whether to allow HTML in the tooltip content
        
        animation: 'scale',     // animation type: scale shift-away
        inertia: true,          // animation inertia
        // arrow: false,           // whether to allow arrows

        // animateFill: true,      // determines if the background fill color should be animated

        // delay: [400, null],     // delay: [show, hide], show delay is 400ms, hide delay is the default        
    }
});
*/

/* 
    Hook System of Tooltip
    The hook system allows you to execute custom logic at specific times, 
    such as adding custom interactions before the tooltip initialization and after initialization.
*/

/*
DocumentDates.registerHook('beforeInit', async (context) => {
    
});

DocumentDates.registerHook('afterInit', async (context) => {
    
});
*/



/*
    Demonstrates how to register a local language when using timeago.js
    Simply translate the English in parentheses into your own language, nothing else needs to be changed!
*/
/*
const localeFunc = (number, index) => {
    return [
        ['just now', 'right now'],
        ['%s seconds ago', 'in %s seconds'],
        ['1 minute ago', 'in 1 minute'],
        ['%s minutes ago', 'in %s minutes'],
        ['1 hour ago', 'in 1 hour'],
        ['%s hours ago', 'in %s hours'],
        ['1 day ago', 'in 1 day'],
        ['%s days ago', 'in %s days'],
        ['1 week ago', 'in 1 week'],
        ['%s weeks ago', 'in %s weeks'],
        ['1 month ago', 'in 1 month'],
        ['%s months ago', 'in %s months'],
        ['1 year ago', 'in 1 year'],
        ['%s years ago', 'in %s years']
    ][index];
};
const localeStr = 'whatever';
timeago.register(localeStr, localeFunc);

if (typeof timeago !== 'undefined') {
    document.querySelectorAll('.document-dates-plugin time').forEach(timeElement => {
        timeElement.textContent = timeago.format(timeElement.getAttribute('datetime'), localeStr);
    });
}
*/