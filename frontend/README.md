# LLM Meta-Analysis Dashboard

## Single-File Deployment

The dashboard is now available as a single HTML file with embedded JavaScript. No build step required!

### Quick Start

Simply open `dashboard.html` in a web browser:

```bash
# On Windows
start dashboard.html

# On macOS
open dashboard.html

# On Linux
xdg-open dashboard.html
```

Or serve it with a local web server:

```bash
# Python 3
python -m http.server 8000

# Node.js (with npx)
npx serve .

# PHP
php -S localhost:8000
```

Then open http://localhost:8000/dashboard.html

## Features

### Dashboard
- Overview statistics
- Quick start guide
- Feature list

### Data Extraction
- Drag-and-drop file upload (PDF, TXT, CSV, JSON)
- Study management
- LLM-powered extraction configuration
  - Outcome type selection (binary, continuous, survival)
  - Model selection (GPT-4, Claude 3, fine-tuned)
  - Advanced features (RAG, Chain-of-Thought, Ensemble)
- Extraction results table

### Meta-Analysis
- Configuration panel:
  - Outcome type (binary, continuous)
  - Effect measure (OR, RR, RD, MD, SMD)
  - Model type (fixed, random effects)
  - CI method (Wald, HKSJ, Profile Likelihood, Bootstrap)
- Interactive plots:
  - Forest Plot
  - Funnel Plot
  - Network Plot

### Reports
- PRISMA 2020 flow diagram
- Risk of Bias assessment (RoB 2)
- Export options (PDF, CSV, JSON)
- Summary recommendations

## WASM Module (Optional)

For faster statistical computations, a WebAssembly module is included.

### Building WASM

Prerequisites:
- Rust toolchain (https://rustup.rs/)
- wasm-pack: `cargo install wasm-pack`

Build:
```bash
cd wasm
./build.sh        # Linux/macOS
# or
build.bat         # Windows
```

This generates:
- `meta_analysis_wasm_bg.wasm` - The WASM binary
- `meta_analysis_wasm.js` - JavaScript bindings

### Using WASM in Dashboard

Add to `dashboard.html` before the closing `</body>` tag:

```html
<script type="module">
    import init, { Study, random_effects_ma } from './meta_analysis_wasm.js';

    async function runAnalysis() {
        const wasm = await init();

        const studies = [
            new Study("Study A", 0.75, 0.55, 1.02, 0.02),
            new Study("Study B", 0.68, 0.48, 0.96, 0.015),
            new Study("Study C", 0.82, 0.60, 1.12, 0.025),
        ];

        const result = random_effects_ma(studies);
        console.log(result.to_json());
    }

    runAnalysis();
</script>
```

### WASM API

#### Classes
- `Study(name, effect, ci_lower, ci_upper, variance)` - Create a study object

#### Functions
- `fixed_effect_ma(studies: Study[])` - Fixed effect meta-analysis
- `random_effects_ma(studies: Study[])` - Random effects (DerSimonian-Laird)
- `hksj_adjustment(studies: Study[])` - Hartung-Knapp-Sidik-Jonkman

#### Result
Returns `MetaAnalysisResult` with:
- `pooled_effect` - Pooled effect estimate
- `ci_lower`, `ci_upper` - 95% confidence interval
- `p_value` - P-value
- `i_squared` - Heterogeneity (I²)
- `tau_squared` - Between-study variance
- `q_statistic` - Cochran's Q
- `df` - Degrees of freedom
- `to_json()` - JSON string representation

## API Configuration

The dashboard connects to a backend API. Configure the API URL:

```javascript
// In dashboard.html, find the API object:
const API = {
    baseUrl: window.location.hostname === 'localhost'
        ? 'http://localhost:8000'
        : window.location.origin,
    // ...
};
```

## CDN Dependencies

The dashboard uses these CDN libraries:

- React 18 - UI framework
- Babel Standalone - JSX compilation
- Plotly.js 2.27 - Interactive plots
- Bootstrap 5.3 - Styling
- Bootstrap Icons - Icons

All loaded automatically from unpkg.com and jsdelivr.net.

## Customization

### Styling
Edit the `<style>` section in `dashboard.html` to customize colors, fonts, etc.

```css
:root {
    --primary-color: #1976d2;
    --secondary-color: #dc004e;
    /* ... */
}
```

### Adding Pages
Add a new page in the `renderPage()` function:

```javascript
case 'mypage':
    return <MyPage />;
```

### Modifying Components
All React components are defined within the same `<script type="text/babel">` tag.

## Browser Compatibility

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+

## Security Notes

- For production, serve over HTTPS
- Validate and sanitize all API responses
- Implement proper authentication
- Use CSP headers to restrict script sources

## Performance Tips

- Enable WASM for faster computations
- Use a CDN for static assets
- Enable gzip compression
- Cache API responses

## Troubleshooting

**API Connection Error**
- Ensure backend is running on the configured port
- Check CORS settings on the backend
- Verify API URL in the API configuration

**Plots Not Displaying**
- Check browser console for errors
- Verify Plotly.js is loading from CDN
- Ensure data is in correct format

**File Upload Not Working**
- Check browser console for errors
- Verify file sizes are within limits
- Ensure correct file types are selected

## License

Same as the main LLM Meta-Analysis Framework project.
