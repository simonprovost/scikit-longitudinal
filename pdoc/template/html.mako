<%
  import os

  import pdoc
  from pdoc.html_helpers import extract_toc, glimpse, to_html as _to_html, format_git_link


  def link(dobj: pdoc.Doc, name=None):
    name = name or dobj.qualname + ('()' if isinstance(dobj, pdoc.Function) else '')
    if isinstance(dobj, pdoc.External) and not external_links:
        return name
    url = dobj.url(relative_to=module, link_prefix=link_prefix,
                   top_ancestor=not show_inherited_members)
    return f'<a title="{dobj.refname}" href="{url}">{name}</a>'


  def to_html(text):
    return _to_html(text, docformat=docformat, module=module, link=link, latex_math=latex_math)


  def get_annotation(bound_method, sep=':'):
    annot = show_type_annotations and bound_method(link=link) or ''
    if annot:
        annot = ' ' + sep + '\N{NBSP}' + annot
    return annot
%>

<%def name="ident(name)"><span class="ident">${name}</span></%def>

<%def name="show_source(d)">
  % if (show_source_code or git_link_template) and d.source and d.obj is not getattr(d.inherits, 'obj', None):
    <% git_link = format_git_link(git_link_template, d) %>
    % if show_source_code:
      <details class="source">
        <summary>
            <span>Expand source code</span>
            % if git_link:
              <a href="${git_link}" class="git-link">Browse git</a>
            %endif
        </summary>
        <pre><code class="python">${d.source | h}</code></pre>
      </details>
    % elif git_link:
      <div class="git-link-div"><a href="${git_link}" class="git-link">Browse git</a></div>
    %endif
  %endif
</%def>

<%def name="show_desc(d, short=False)">
  <%
  inherits = ' inherited' if d.inherits else ''
  docstring = glimpse(d.docstring) if short or inherits else d.docstring
  %>
  % if d.inherits:
      <p class="inheritance">
          <em>Inherited from:</em>
          % if hasattr(d.inherits, 'cls'):
              <code>${link(d.inherits.cls)}</code>.<code>${link(d.inherits, d.name)}</code>
          % else:
              <code>${link(d.inherits)}</code>
          % endif
      </p>
  % endif
  <div class="desc${inherits}">${docstring | to_html}</div>
  % if not isinstance(d, pdoc.Module):
  ${show_source(d)}
  % endif
</%def>

<%def name="show_module_list(modules)">
<h1>Python module list</h1>

% if not modules:
  <p>No modules found.</p>
% else:
  <dl id="http-server-module-list">
  % for name, desc in modules:
      <div class="flex">
      <dt><a href="${link_prefix}${name}">${name}</a></dt>
      <dd>${desc | glimpse, to_html}</dd>
      </div>
  % endfor
  </dl>
% endif
</%def>

<%def name="show_column_list(items)">
  <%
      two_column = len(items) >= 6 and all(len(i.name) < 20 for i in items)
  %>
  <ul class="${'two-column' if two_column else ''}">
  % for item in items:
    <li><code>${link(item, item.name)}</code></li>
  % endfor
  </ul>
</%def>

<%def name="show_module(module)">
  <%
  variables = module.variables(sort=sort_identifiers)
  classes = module.classes(sort=sort_identifiers)
  functions = module.functions(sort=sort_identifiers)
  submodules = module.submodules()
  %>

  <%def name="show_func(f)">
    <dt id="${f.refname}"><code class="name flex">
        <%
            params = ', '.join(f.params(annotate=show_type_annotations, link=link))
            return_type = get_annotation(f.return_annotation, '\N{non-breaking hyphen}>')
        %>
        <span>${f.funcdef()} ${ident(f.name)}</span>(<span>${params})${return_type}</span>
    </code></dt>
    <dd>${show_desc(f)}</dd>
  </%def>

  <header>
  % if http_server:
    <nav class="http-server-breadcrumbs">
      <a href="/">All packages</a>
      <% parts = module.name.split('.')[:-1] %>
      % for i, m in enumerate(parts):
        <% parent = '.'.join(parts[:i+1]) %>
        :: <a href="/${parent.replace('.', '/')}/">${parent}</a>
      % endfor
    </nav>
  % endif
  % if module.name != "scikit_longitudinal":
  <h1 class="title">${'Namespace' if module.is_namespace else  \
                      'Package' if module.is_package and not module.supermodule else \
                      'Module'} <code>${module.name}</code></h1>
  % endif
  </header>

  <section id="section-intro">
  ${module.docstring | to_html}
  ${show_source(module)}
  </section>

  <section>
    % if submodules:
    % if module.name != "scikit_longitudinal":
    <h2 class="section-title" id="header-submodules">Sub-modules</h2>
    % endif
    <dl>
    % for m in submodules:
      <dt><code class="name">${ link(m, name=m.name.split('.')[-1]) }</code></dt>
      <dd>${show_desc(m, short=True)}</dd>
    % endfor
    </dl>
    % endif
  </section>

  <section>
    % if variables:
    <h2 class="section-title" id="header-variables">Global variables</h2>
    <dl>
    % for v in variables:
      <% return_type = get_annotation(v.type_annotation) %>
      <dt id="${v.refname}"><code class="name">var ${ident(v.name)}${return_type}</code></dt>
      <dd>${show_desc(v)}</dd>
    % endfor
    </dl>
    % endif
  </section>

  <section>
    % if functions:
    <h2 class="section-title" id="header-functions">Functions</h2>
    <dl>
    % for f in functions:
      ${show_func(f)}
    % endfor
    </dl>
    % endif
  </section>

  <section>
    % if classes:
    <h2 class="section-title" id="header-classes">Classes</h2>
    <dl>
    % for c in classes:
      <%
      class_vars = c.class_variables(show_inherited_members, sort=sort_identifiers)
      smethods = c.functions(show_inherited_members, sort=sort_identifiers)
      inst_vars = c.instance_variables(show_inherited_members, sort=sort_identifiers)
      methods = c.methods(show_inherited_members, sort=sort_identifiers)
      mro = c.mro()
      subclasses = c.subclasses()
      params = ', '.join(c.params(annotate=show_type_annotations, link=link))
      %>
      <dt id="${c.refname}"><code class="flex name class">
          <span>class ${ident(c.name)}</span>
          % if params:
              <span>(</span><span>${params})</span>
          % endif
      </code></dt>

      <dd>${show_desc(c)}

      % if subclasses:
          <h3>Subclasses</h3>
          <ul class="hlist">
          % for sub in subclasses:
              <li>${link(sub)}</li>
          % endfor
          </ul>
      % endif
      % if class_vars:
          <h3>Class variables</h3>
          <dl>
          % for v in class_vars:
              <% return_type = get_annotation(v.type_annotation) %>
              <dt id="${v.refname}"><code class="name">var ${ident(v.name)}${return_type}</code></dt>
              <dd>${show_desc(v)}</dd>
          % endfor
          </dl>
      % endif
      % if smethods:
          <h3>Static methods</h3>
          <dl>
          % for f in smethods:
              ${show_func(f)}
          % endfor
          </dl>
      % endif
      % if inst_vars:
          <h3>Instance variables</h3>
          <dl>
          % for v in inst_vars:
              <% return_type = get_annotation(v.type_annotation) %>
              <dt id="${v.refname}"><code class="name">var ${ident(v.name)}${return_type}</code></dt>
              <dd>${show_desc(v)}</dd>
          % endfor
          </dl>
      % endif
      % if methods:
          <h3>Methods</h3>
          <dl>
          % for f in methods:
              ${show_func(f)}
          % endfor
          </dl>
      % endif

      % if not show_inherited_members:
          <%
              members = c.inherited_members()
          %>
          % if members:
              <h3>Inherited members</h3>
              <ul class="hlist">
              % for cls, mems in members:
                  <li><code><b>${link(cls)}</b></code>:
                      <ul class="hlist">
                          % for m in mems:
                              <li><code>${link(m, name=m.name)}</code></li>
                          % endfor
                      </ul>

                  </li>
              % endfor
              </ul>
          % endif
      % endif

      </dd>
    % endfor
    </dl>
    % endif
  </section>
</%def>

<%def name="module_index(module)">
  <%
  variables = module.variables(sort=sort_identifiers)
  classes = module.classes(sort=sort_identifiers)
  functions = module.functions(sort=sort_identifiers)
  submodules = module.submodules()
  supermodule = module.supermodule
  %>
  <nav id="sidebar">

    <%include file="logo.mako"/>

    % if google_search_query:
        <div class="gcse-search" style="height: 70px"
             data-as_oq="${' '.join(google_search_query.strip().split()) | h }"
             data-gaCategoryParameter="${module.refname | h}">
        </div>
    % endif

    % if lunr_search is not None:
      <%include file="_lunr_search.inc.mako"/>
    % endif

    <h1>Index</h1>
    ${extract_toc(module.docstring) if extract_module_toc_into_sidebar else ''}
    <ul id="index">
    % if supermodule:
    <li><h3>Super-module</h3>
      <ul>
        <li><code>${link(supermodule)}</code></li>
      </ul>
    </li>
    % endif

    % if submodules:
    <li><h3><a href="#header-submodules">Sub-modules</a></h3>
      <ul>
      % for m in submodules:
        <li><code>${ link(m, name=m.name.split('.')[-1]) }</code></li>
      % endfor
      </ul>
    </li>
    % endif

    % if variables:
    <li><h3><a href="#header-variables">Global variables</a></h3>
      ${show_column_list(variables)}
    </li>
    % endif

    % if functions:
    <li><h3><a href="#header-functions">Functions</a></h3>
      ${show_column_list(functions)}
    </li>
    % endif

    % if classes:
    <li><h3><a href="#header-classes">Classes</a></h3>
      <ul>
      % for c in classes:
        <li>
        <h4><code>${link(c)}</code></h4>
        <%
            members = c.functions(sort=sort_identifiers) + c.methods(sort=sort_identifiers)
            if list_class_variables_in_index:
                members += (c.instance_variables(sort=sort_identifiers) +
                            c.class_variables(sort=sort_identifiers))
            if not show_inherited_members:
                members = [i for i in members if not i.inherits]
            if sort_identifiers:
              members = sorted(members)
        %>
        % if members:
          ${show_column_list(members)}
        % endif
        </li>
      % endfor
      </ul>
    </li>
    % endif

    </ul>
  </nav>
</%def>

<!doctype html>
<html lang="${html_lang}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, minimum-scale=1" />
  <meta name="generator" content="pdoc ${pdoc.__version__}" />

<%
    module_list = 'modules' in context.keys()  # Whether we're showing module list in server mode
%>

  % if module_list:
    <title>Python module list</title>
    <meta name="description" content="A list of documented Python modules." />
  % else:
    <title>${module.name} API documentation</title>
    <meta name="description" content="${module.docstring | glimpse, trim, h}" />
  % endif

  <link rel="preload stylesheet" as="style" href="https://cdnjs.cloudflare.com/ajax/libs/10up-sanitize.css/11.0.1/sanitize.min.css" integrity="sha256-PK9q560IAAa6WVRRh76LtCaI8pjTJ2z11v0miyNNjrs=" crossorigin>
  <link rel="preload stylesheet" as="style" href="https://cdnjs.cloudflare.com/ajax/libs/10up-sanitize.css/11.0.1/typography.min.css" integrity="sha256-7l/o7C8jubJiy74VsKTidCy1yBkRtiUGbVkYBylBqUg=" crossorigin>
  % if syntax_highlighting:
    <link rel="stylesheet preload" as="style" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/10.1.1/styles/${hljs_style}.min.css" crossorigin>
  %endif

  <%namespace name="css" file="css.mako" />
  <style>${css.mobile()}</style>
  <style media="screen and (min-width: 700px)">${css.desktop()}</style>
  <style media="print">${css.print()}</style>

  % if google_analytics:
    <script>
    window.ga=window.ga||function(){(ga.q=ga.q||[]).push(arguments)};ga.l=+new Date;
    ga('create', '${google_analytics}', 'auto'); ga('send', 'pageview');
    </script><script async src='https://www.google-analytics.com/analytics.js'></script>
  % endif

  % if google_search_query:
    <link rel="preconnect" href="https://www.google.com">
    <script async src="https://cse.google.com/cse.js?cx=017837193012385208679:pey8ky8gdqw"></script>
    <style>
        .gsc-control-cse {padding:0 !important;margin-top:1em}
        body.gsc-overflow-hidden #sidebar {overflow: visible;}
    </style>
  % endif

  % if latex_math:
    <script async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-AMS_CHTML" integrity="sha256-kZafAc6mZvK3W3v1pHOcUix30OHQN6pU/NO2oFkqZVw=" crossorigin></script>
  % endif

  % if syntax_highlighting:
    <script defer src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/10.1.1/highlight.min.js" integrity="sha256-Uv3H6lx7dJmRfRvH8TH6kJD1TSK1aFcwgx+mdg3epi8=" crossorigin></script>
    <script>window.addEventListener('DOMContentLoaded', () => hljs.initHighlighting())</script>
  % endif

  % if module.name == "scikit_longitudinal":
  <style>
    #sidebar { display: none; }
    #content { margin: auto; width: 100%; border-left: none; }
  </style>
  % endif


  <%include file="head.mako"/>
</head>
<body>
<main>
  % if module_list:
    <article id="content">
      ${show_module_list(modules)}
    </article>
  % else:
    <article id="content">
      % if module.name == "scikit_longitudinal":
        <section id="welcome-message">
          <!--suppress HtmlDeprecatedAttribute -->
          <div align="center">
            <h1 align="center">
              <br>
              <img src="./logo.png" alt="Scikit-longitudinal" width="200">
              <br>
              Scikit-longitudinal
              <br>
            </h1>
            <h4 align="center">Scikit-longitudinal is a <b>machine learning library</b> designed specifically for the analysis
            of <b>longitudinal data</b>. It provides a collection of tools and models for processing,
            analysing, and making predictions on longitudinal data, with an <b>easy-to-use interface</b>
            that is compatible with the <b>Scikit-learn ecosystem</b>.</h4>
            <table align="center">
              <tr>
         <td align="center">
            <h3>⚙️ Project Status</h3>
         </td>
         <td align="center">
            <h3>☎️ Contacts</h3>
         </td>
      </tr>
      <tr>
         <td valign="top">
            <!-- Python-related badges table -->
            <table>
               <tr>
                  <table>
                     <tr>
                        <td>
                           <a href="https://pdm.fming.dev">
                           <img alt="pdm" src="https://img.shields.io/badge/pdm-managed-blue?style=for-the-badge&logo=python">
                           </a>
                        </td>
                        <td>
                           <a href="https://pytest.org/">
                           <img alt="pytest" src="https://img.shields.io/badge/pytest-passing-green?style=for-the-badge&logo=pytest">
                           </a><br />
                           <a href="https://codecov.io/gh/Scikit-Longitudinal/Scikit-Longitudinal">
                           <img alt="Codecov" src="https://img.shields.io/badge/coverage-85%25-brightgreen.svg?style=for-the-badge&logo=appveyor">
                           </a>
                        </td>
                     </tr>
                     <tr>
                        <td>
                           <a href="https://flake8.pycqa.org/en/latest/">
                           <img alt="flake8" src="https://img.shields.io/badge/flake8-checked-blue?style=for-the-badge&logo=python">
                           </a><br />
                           <a href="https://www.pylint.org/">
                           <img alt="pylint" src="https://img.shields.io/badge/pylint-checked-blue?style=for-the-badge&logo=python">
                           </a><br />
                           <a href="https://pre-commit.com/">
                           <img alt="pre-commit" src="https://img.shields.io/badge/pre--commit-checked-blue?style=for-the-badge&logo=python">
                           </a>
                        </td>
                        <td>
                           <a href="https://github.com/PyCQA/isort">
                           <img alt="isort" src="https://img.shields.io/badge/isort-compliant-green?style=for-the-badge&logo=python">
                           </a><br />
                           <a href="https://github.com/psf/black">
                           <img alt="black" src="https://img.shields.io/badge/black-formatted-black?style=for-the-badge&logo=python">
                           </a><br />
                           <a href="https://github.com/hhatto/autopep8">
                           <img alt="autopep8" src="https://img.shields.io/badge/autopep8-compliant-green?style=for-the-badge&logo=python">
                           </a>
                        </td>
                     </tr>
                  </table>
                  <td valign="center">
                     <table>
                        <tr>
                           <td>
                              <a href="mailto:sgp28@kent.ac.uk">
                              <img alt="Microsoft Outlook" src="https://img.shields.io/badge/Microsoft_Outlook-0078D4?style=for-the-badge&logo=microsoft-outlook&logoColor=white">
                              </a><br />
                              <a href="https://linkedin.com/in/simonprovostdev/">
                              <img alt="LinkedIn" src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white">
                              </a><br />
                              <a href="https://discord.com/users/Simon__#6384">
                              <img alt="Discord" src="https://img.shields.io/badge/Discord-7289DA?style=for-the-badge&logo=discord&logoColor=white">
                              </a><br />
                            <a href="https://stackoverflow.com/users/9814037/simon-provost">
                              <img alt="Stack Overflow" src="https://img.shields.io/badge/Stack_Overflow-FE7A16?style=for-the-badge&logo=stack-overflow&logoColor=white">
                              </a><br />
                              <a href="https://twitter.com/SimonProvost_">
                              <img alt="Twitter" src="https://img.shields.io/badge/Twitter-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white">
                              </a><br />
                              <a href="https://scholar.google.com/citations?user=Lv_LddYAAAAJ&hl=en&authuser=3">
                              <img alt="Google Scholar" src="https://img.shields.io/badge/Google_Scholar-4285F4?style=for-the-badge&logo=google-scholar&logoColor=white">
                              </a>
                           </td>
                        </tr>
                     </table>
                  </td>
               </tr>
            </table>
          </div>
          <!-- TABLE OF CONTENTS -->
          <summary style="font-size: 1.5em; font-weight: bold;">
            📚 What the library is composed of ?
          </summary>
          <ol>
            ${show_module(module)}
          </ol>
        <section id="additional-info">
          <h2 style="font-size: 1.5em; font-weight: bold;">🎙️Additional Information</h2>
          <details>
            <summary style="font-size: 1.2em;">
              💡How does the temporal information is handled/encoded ?
            </summary>
            <p>
              In the <a href="https://github.com/simonprovost/scikit-longitudinal/blob/feature/general_documentation/scikit_longitudinal/README.md">Readme.md</a>  of the source files (not the root) we described the method we used for a generic handling of temporal information.
              Keep in mind that as a developer or designer, you, broadly speaking, have access to a new matrix containing time information according to longitudinal attributes.
              Therefore, when dealing with a particular feature of your dataset, you have access to its corresponding time information (wave).
            </p>
          </details>
          <details>
            <summary style="font-size: 1.2em;">
              🐞 How to Report Bugs or Request Features
            </summary>
            <p>
              To report bugs, request features, or ask questions, please use the GitHub issues section of the project.
              <a href="https://github.com/simonprovost/scikit-longitudinal/issues/new/choose">Click here to go to GitHub Issues.</a>
            </p>
          </details>
          <details>
            <summary style="font-size: 1.2em;">
              💻 Developer Setup
            </summary>
            <p>
              For setting up the project as a developer,
              <a href="https://github.com/simonprovost/scikit-longitudinal/blob/main/README.md#-prerequisites">refer to this section of the README.</a>
            </p>
          </details>
          <details>
            <summary style="font-size: 1.2em;">
              🧪 Experiments
            </summary>
            <p>
              To explore the experiments on each reproduced algorithms from the literature, please switch to the specific branch called <a href="https://github.com/simonprovost/scikit-longitudinal/tree/experiments">Experiments</a> and navigate to the "experiments" folder.
              A Makefile is included for convenience.
            </p>
          </details>
          <details>
            <summary style="font-size: 1.2em;">
              📝 Citing Scikit-Longitudinal
            </summary>
            <p>
              This section is a placeholder. Citation information will be available once papers are published.
            </p>
          </details>
        </section>
          <!-- About the project section goes here as in README -->
        </section>
      % endif
      % if module.name != "scikit_longitudinal":
        ${show_module(module)}
      % endif
    </article>
    % if module.name != "scikit_longitudinal":
    ${module_index(module)}
    % endif
  % endif
</main>



<footer id="footer">
    <%include file="credits.mako"/>
    <p>Generated by <a href="https://pdoc3.github.io/pdoc" title="pdoc: Python API documentation generator"><cite>pdoc</cite> ${pdoc.__version__}</a>.</p>
</footer>

% if http_server and module:  ## Auto-reload on file change in dev mode
    <script>
    setInterval(() =>
        fetch(window.location.href, {
            method: "HEAD",
            cache: "no-store",
            headers: {"If-None-Match": "${os.stat(module.obj.__file__).st_mtime}"},
        }).then(response => response.ok && window.location.reload()), 700);
    </script>
% endif
</body>
</html>
